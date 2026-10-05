//! Teachers: models that judge a request's questions and return a probability per
//! option. A teacher is any `/v1/systemone` server ([`HttpTeacher`]), so the large
//! model behind the training is swapped by pointing at another server, on this
//! machine or another; several teachers combine into one ([`Committee`]), and every
//! judgement is kept on disk ([`Cache`]) so a restarted run asks nothing twice.

use anyhow::{bail, ensure, Context, Result};
use ojas_decision::json::Json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// Per question, the probability of each option keyed as the request names it: a
/// choice's keys, `"0"`… for a score's levels, `"false"` / `"true"` for a noul.
pub type Judgement = Vec<HashMap<String, f64>>;

pub trait Teacher: Send + Sync {
    fn name(&self) -> &str;
    fn judge(&self, body: &Json) -> Result<Judgement>;
}

/// A `/v1/systemone` server.
pub struct HttpTeacher {
    name: String,
    host: String,
    port: u16,
    path: String,
}

impl HttpTeacher {
    /// `url` is `http://host:port[/path]`; the path defaults to `/v1/systemone`.
    pub fn new(name: &str, url: &str) -> Result<Self> {
        let u = url.trim_end_matches('/').trim_start_matches("http://");
        let (hostport, path) = u.split_once('/').map(|(h, p)| (h, format!("/{p}"))).unwrap_or((u, "/v1/systemone".into()));
        let (host, port) = hostport.split_once(':').with_context(|| format!("{url}: expected host:port"))?;
        Ok(HttpTeacher { name: name.to_string(), host: host.to_string(), port: port.parse().with_context(|| format!("{url}: bad port"))?, path })
    }

    /// One request, tried up to three times with a pause between attempts, so a
    /// teacher that is restarting or momentarily overloaded does not end a run.
    fn post(&self, body: &str) -> Result<String> {
        let mut last = None;
        for attempt in 0..3 {
            if attempt > 0 { std::thread::sleep(Duration::from_secs(5 << attempt)); }
            match self.post_once(body) {
                Ok(text) => return Ok(text),
                Err(e) => last = Some(e),
            }
        }
        Err(last.expect("three attempts"))
    }

    fn post_once(&self, body: &str) -> Result<String> {
        let mut s = TcpStream::connect((self.host.as_str(), self.port))
            .with_context(|| format!("connecting to teacher {} at {}:{}", self.name, self.host, self.port))?;
        s.set_read_timeout(Some(Duration::from_secs(900)))?;
        write!(s, "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
               self.path, self.host, body.len())?;
        let mut r = BufReader::new(s);
        let mut status = String::new();
        r.read_line(&mut status)?;
        let (mut length, mut chunked) = (None, false);
        loop {
            let mut line = String::new();
            r.read_line(&mut line)?;
            let line = line.trim_end();
            if line.is_empty() { break; }
            if let Some((k, v)) = line.split_once(':') {
                match k.to_ascii_lowercase().as_str() {
                    "content-length" => length = v.trim().parse::<usize>().ok(),
                    "transfer-encoding" => chunked = v.to_ascii_lowercase().contains("chunked"),
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        if chunked {
            loop {
                let mut size = String::new();
                r.read_line(&mut size)?;
                let n = usize::from_str_radix(size.split(';').next().unwrap_or("").trim(), 16).context("bad chunk size")?;
                if n == 0 { break; }
                let mut chunk = vec![0u8; n + 2];
                r.read_exact(&mut chunk)?;
                out.extend_from_slice(&chunk[..n]);
            }
        } else if let Some(n) = length {
            out.resize(n, 0);
            r.read_exact(&mut out)?;
        } else {
            r.read_to_end(&mut out)?;
        }
        let text = String::from_utf8_lossy(&out).into_owned();
        ensure!(status.split_whitespace().nth(1) == Some("200"), "teacher {}: {}: {}", self.name, status.trim(), text.chars().take(200).collect::<String>());
        Ok(text)
    }
}

impl Teacher for HttpTeacher {
    fn name(&self) -> &str { &self.name }

    fn judge(&self, body: &Json) -> Result<Judgement> {
        let text = self.post(&body.to_python(false))?;
        let response = Json::parse(&text).map_err(|e| anyhow::anyhow!("teacher {}: {e}", self.name))?;
        let questions = body.get("questions").and_then(Json::as_object).context("a request has questions")?;
        let answers = response.get("answers").context("the response has no answers")?;
        questions.iter().map(|(id, q)| {
            let a = answers.get(id).with_context(|| format!("teacher {} answered no question {id}", self.name))?;
            let mut p = HashMap::new();
            match q.get("type").and_then(Json::as_str) {
                Some("noul") => {
                    let t = a.get("noul").and_then(Json::as_f64).context("a noul answer")?;
                    p.insert("true".into(), t);
                    p.insert("false".into(), 1.0 - t);
                }
                _ => {
                    let probs = a.get("probabilities").and_then(Json::as_object).context("an answer's probabilities")?;
                    for (k, v) in probs { p.insert(k.clone(), v.as_f64().unwrap_or(0.0)); }
                }
            }
            Ok(p)
        }).collect()
    }
}

/// Several teachers as one: the weighted mean of their probabilities.
pub struct Committee {
    name: String,
    members: Vec<(Box<dyn Teacher>, f64)>,
}

impl Committee {
    pub fn new(members: Vec<(Box<dyn Teacher>, f64)>) -> Result<Self> {
        ensure!(!members.is_empty(), "a committee needs a teacher");
        ensure!(members.iter().all(|(_, w)| *w > 0.0), "committee weights are positive");
        let name = members.iter().map(|(t, w)| format!("{}:{w}", t.name())).collect::<Vec<_>>().join("+");
        Ok(Committee { name, members })
    }
}

impl Teacher for Committee {
    fn name(&self) -> &str { &self.name }

    fn judge(&self, body: &Json) -> Result<Judgement> {
        let total: f64 = self.members.iter().map(|(_, w)| w).sum();
        let mut merged: Option<Judgement> = None;
        for (teacher, weight) in &self.members {
            let j = teacher.judge(body)?;
            match &mut merged {
                None => merged = Some(j.into_iter().map(|q| q.into_iter().map(|(k, p)| (k, p * weight / total)).collect()).collect()),
                Some(m) => {
                    ensure!(m.len() == j.len(), "teachers answered different numbers of questions");
                    for (acc, q) in m.iter_mut().zip(j) {
                        for (k, p) in q { *acc.entry(k).or_insert(0.0) += p * weight / total; }
                    }
                }
            }
        }
        Ok(merged.expect("at least one member"))
    }
}

/// Judgements kept in a JSONL file, keyed by teacher and request, and served from
/// memory once read.
pub struct Cache {
    inner: Box<dyn Teacher>,
    path: PathBuf,
    known: Mutex<HashMap<String, Judgement>>,
}

impl Cache {
    pub fn open(inner: Box<dyn Teacher>, path: &Path) -> Result<Self> {
        // A line that does not parse is a judgement lost, not a file lost: it is
        // skipped and counted, and the teacher is asked again when it comes up.
        let mut known = HashMap::new();
        let mut unreadable = 0;
        if path.is_file() {
            for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()) {
                let parsed = Json::parse(line).ok().and_then(|row| {
                    let key = row.get("key").and_then(Json::as_str)?.to_string();
                    let judgement = row.get("judgement").and_then(Json::as_array)?;
                    let j: Judgement = judgement.iter().map(|q| q.as_object().unwrap_or(&[]).iter()
                        .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0))).collect()).collect();
                    Some((key, j))
                });
                match parsed {
                    Some((key, j)) => { known.insert(key, j); }
                    None => unreadable += 1,
                }
            }
        }
        if unreadable > 0 {
            eprintln!("{}: {unreadable} unreadable lines skipped", path.display());
        }
        Ok(Cache { inner, path: path.to_path_buf(), known: Mutex::new(known) })
    }

    fn key(&self, body: &Json) -> String {
        let mut h = Sha256::new();
        h.update(self.inner.name().as_bytes());
        h.update(b"\n");
        h.update(body.sorted().to_python(true).as_bytes());
        format!("{:x}", h.finalize())
    }

    pub fn len(&self) -> usize { self.known.lock().unwrap().len() }

    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl Teacher for Cache {
    fn name(&self) -> &str { self.inner.name() }

    fn judge(&self, body: &Json) -> Result<Judgement> {
        let key = self.key(body);
        if let Some(j) = self.known.lock().unwrap().get(&key) { return Ok(j.clone()); }
        let j = self.inner.judge(body)?;
        let row = Json::Object(vec![
            ("key".into(), Json::Str(key.clone())),
            ("judgement".into(), Json::Array(j.iter().map(|q| Json::Object(q.iter().map(|(k, p)| (k.clone(), Json::Float(*p))).collect())).collect())),
        ]);
        // One write of the whole line, under the lock: threads judging at once neither
        // interleave their lines nor record the same key twice.
        let mut known = self.known.lock().unwrap();
        if let std::collections::hash_map::Entry::Vacant(slot) = known.entry(key) {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)
                .with_context(|| format!("opening {}", self.path.display()))?;
            f.write_all(format!("{}\n", row.to_python(true)).as_bytes())?;
            slot.insert(j.clone());
        }
        Ok(j)
    }
}

/// `name=url[:weight]` specs into one teacher: one spec is that server, several are a
/// committee. A weight follows the url after a colon only when it parses as a number.
pub fn from_specs(specs: &[String]) -> Result<Box<dyn Teacher>> {
    let mut members: Vec<(Box<dyn Teacher>, f64)> = Vec::new();
    for spec in specs {
        let (name, rest) = spec.split_once('=').with_context(|| format!("teacher spec {spec:?}: expected name=url"))?;
        // A trailing `:weight` is one colon past the port, so it is read only when the
        // address before it still has one, with the scheme set aside.
        let bare = rest.trim_start_matches("http://");
        let (url, weight) = match bare.rsplit_once(':') {
            Some((u, w)) if u.contains(':') && w.parse::<f64>().is_ok() => (&rest[..rest.len() - w.len() - 1], w.parse::<f64>().unwrap()),
            _ => (rest, 1.0),
        };
        members.push((Box::new(HttpTeacher::new(name, url)?), weight));
    }
    if members.is_empty() { bail!("no teacher given"); }
    if members.len() == 1 { return Ok(members.pop().unwrap().0); }
    Ok(Box::new(Committee::new(members)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_reads_its_weight_only_past_the_port() {
        assert_eq!(from_specs(&["a=http://127.0.0.1:8080".into()]).unwrap().name(), "a");
        assert_eq!(from_specs(&["a=http://127.0.0.1:8080".into(), "b=10.0.0.5:8080:0.5".into()]).unwrap().name(), "a:1+b:0.5");
        assert!(from_specs(&["a=http://127.0.0.1".into()]).is_err());
    }
}
