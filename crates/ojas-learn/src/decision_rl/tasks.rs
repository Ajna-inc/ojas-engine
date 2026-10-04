//! Decision tasks made on demand: each family writes a state and typed questions in
//! the systemone request shape, together with the right answer, which the family
//! knows because it built the state. A trainer with no data draws from these without
//! limit; requests whose answer only a model can judge come from a request file
//! ([`FileSource`]) and are labelled by a teacher.
//!
//! Every request can be rewritten without changing its answers ([`variants`]):
//! choice options reordered, object keys shuffled. A model that knows the answer gives
//! the same one to every rewrite, so the rewrites are both more examples and a check.

use anyhow::{Context, Result};
use ojas_decision::json::Json;
use std::path::{Path, PathBuf};

/// A request with, per question, the key of the right option when it is known: a
/// choice's key, `"true"` / `"false"` for a noul, a score's level index as text.
#[derive(Clone, Debug)]
pub struct Task {
    pub family: String,
    pub body: Json,
    pub gold: Vec<Option<String>>,
}

/// The families that know their answers.
pub const FAMILIES: &[&str] = &["compare", "sum_check", "parity", "json_lookup", "json_count", "negation", "order", "unit"];

/// A small deterministic generator (xorshift64*), so a seed reproduces a run.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1) }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize { (self.next_u64() % n.max(1) as u64) as usize }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T { &items[self.below(items.len())] }

    pub fn chance(&mut self, p: f64) -> bool { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 <= p }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

fn s(text: impl Into<String>) -> Json { Json::Str(text.into()) }
fn int(n: i64) -> Json { Json::Int(n.to_string()) }
fn obj(kv: Vec<(&str, Json)>) -> Json { Json::Object(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect()) }

fn choice(instructions: &str, options: &[&str]) -> Json {
    obj(vec![
        ("type", s("choice")), ("instructions", s(instructions)),
        ("criteria", Json::Object(options.iter().map(|k| (k.to_string(), Json::Null)).collect())),
    ])
}

fn noul(instructions: &str) -> Json { obj(vec![("type", s("noul")), ("instructions", s(instructions))]) }

fn score(instructions: &str, levels: &[&str]) -> Json {
    obj(vec![("type", s("score")), ("instructions", s(instructions)), ("criteria", Json::Array(levels.iter().map(|l| s(*l)).collect()))])
}

fn request(state: Json, questions: Vec<(&str, Json)>) -> Json {
    obj(vec![("state", state), ("questions", Json::Object(questions.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))])
}

fn task(family: &str, body: Json, gold: Vec<Option<String>>) -> Task { Task { family: family.to_string(), body, gold } }

fn truth(holds: bool) -> Option<String> { Some(if holds { "true" } else { "false" }.to_string()) }

/// One task of `family`.
pub fn generate(family: &str, rng: &mut Rng) -> Task {
    match family {
        "compare" => compare(rng),
        "sum_check" => sum_check(rng),
        "parity" => parity(rng),
        "json_lookup" => json_lookup(rng),
        "json_count" => json_count(rng),
        "negation" => negation(rng),
        "order" => order(rng),
        "unit" => unit(rng),
        other => panic!("no task family {other}"),
    }
}

fn compare(rng: &mut Rng) -> Task {
    let scale = [10, 100, 1000, 100000][rng.below(4)];
    let (a, b) = (rng.below(scale) as i64, rng.below(scale) as i64);
    if a == b { return compare(rng); }
    let body = request(obj(vec![("a", int(a)), ("b", int(b))]), vec![("larger", choice("Which of the two numbers is larger?", &["a", "b"]))]);
    task("compare", body, vec![Some(if a > b { "a" } else { "b" }.to_string())])
}

fn sum_check(rng: &mut Rng) -> Task {
    let (a, b) = (rng.below(500) as i64, rng.below(500) as i64);
    let correct = rng.chance(0.5);
    let shown = if correct { a + b } else { a + b + [1, -1, 10, -10, 2][rng.below(5)] };
    let body = request(s(format!("{a} + {b} = {shown}")), vec![("correct", noul("Is this equation correct?"))]);
    task("sum_check", body, vec![truth(correct)])
}

fn parity(rng: &mut Rng) -> Task {
    let n = rng.below(100000) as i64;
    let body = request(obj(vec![("number", int(n))]), vec![("even", noul("Is the number even?"))]);
    task("parity", body, vec![truth(n % 2 == 0)])
}

const WORDS: &[&str] = &["apple", "river", "copper", "violet", "harbor", "meadow", "signal", "marble", "falcon", "lantern",
                         "cedar", "quartz", "ember", "tundra", "orbit", "saffron"];
const KEYS: &[&str] = &["name", "city", "color", "status", "owner", "team", "tag", "type", "label", "group"];

fn json_lookup(rng: &mut Rng) -> Task {
    let n = 3 + rng.below(4);
    let mut keys: Vec<&str> = KEYS.to_vec();
    rng.shuffle(&mut keys);
    let mut values: Vec<&str> = WORDS.to_vec();
    rng.shuffle(&mut values);
    let kv: Vec<(&str, &str)> = keys[..n].iter().copied().zip(values[..n].iter().copied()).collect();
    let (ask_key, ask_value) = kv[rng.below(n)];
    let state = Json::Object(kv.iter().map(|(k, v)| (k.to_string(), s(*v))).collect());
    let mut options: Vec<&str> = kv.iter().map(|(_, v)| *v).collect();
    rng.shuffle(&mut options);
    let body = request(state, vec![("value", choice(&format!("What is the value of \"{ask_key}\"?"), &options))]);
    task("json_lookup", body, vec![Some(ask_value.to_string())])
}

fn json_count(rng: &mut Rng) -> Task {
    let n = 1 + rng.below(5);
    let mut values: Vec<&str> = WORDS.to_vec();
    rng.shuffle(&mut values);
    let state = obj(vec![("items", Json::Array(values[..n].iter().map(|v| s(*v)).collect()))]);
    let body = request(state, vec![("count", score("How many items does the list hold?", &["one", "two", "three", "four", "five"]))]);
    task("json_count", body, vec![Some((n - 1).to_string())])
}

const NOUNS: &[&str] = &["door", "light", "engine", "window", "alarm", "printer", "valve", "gate"];
const STATES: &[(&str, &str)] = &[("open", "closed"), ("on", "off"), ("running", "stopped"), ("locked", "unlocked"), ("full", "empty")];

fn negation(rng: &mut Rng) -> Task {
    let noun = *rng.pick(NOUNS);
    let (a, b) = *rng.pick(STATES);
    let (said, other) = if rng.chance(0.5) { (a, b) } else { (b, a) };
    let negated = rng.chance(0.5);
    let state = if negated { format!("The {noun} is not {said}; it is {other}.") } else { format!("The {noun} is {said}.") };
    let asked = if rng.chance(0.5) { said } else { other };
    let holds = (asked == said) != negated;
    let body = request(s(state), vec![("holds", noul(&format!("Is the {noun} {asked}?")))]);
    task("negation", body, vec![truth(holds)])
}

const PEOPLE: &[&str] = &["Ana", "Ben", "Chloe", "Dev", "Esra", "Finn", "Gita", "Hugo"];

fn order(rng: &mut Rng) -> Task {
    let mut people: Vec<&str> = PEOPLE.to_vec();
    rng.shuffle(&mut people);
    let (a, b, c) = (people[0], people[1], people[2]);
    let state = format!("{a} arrived before {b}. {b} arrived before {c}.");
    let (x, y) = if rng.chance(0.5) { (a, c) } else { (c, a) };
    let body = request(s(state), vec![("before", noul(&format!("Did {x} arrive before {y}?")))]);
    task("order", body, vec![truth(x == a)])
}

const UNITS: &[(&str, &str, i64)] = &[("km", "m", 1000), ("kg", "g", 1000), ("hours", "minutes", 60), ("m", "cm", 100)];

fn unit(rng: &mut Rng) -> Task {
    let (big, small, factor) = *rng.pick(UNITS);
    let a = 1 + rng.below(20) as i64;
    let b = a * factor + [factor / 2, -factor / 2, factor, -factor][rng.below(4)] * (rng.chance(0.5) as i64);
    let body = request(obj(vec![("first", s(format!("{a} {big}"))), ("second", s(format!("{b} {small}")))]),
        vec![("larger", choice("Which quantity is larger?", &["first", "second"]))]);
    if a * factor == b { return unit(rng); }
    task("unit", body, vec![Some(if a * factor > b { "first" } else { "second" }.to_string())])
}

/// Rewrites of `body` with the same answers: every choice's options reversed, and
/// the keys of an object state shuffled. Each is a separate request.
pub fn variants(body: &Json, rng: &mut Rng) -> Vec<Json> {
    let mut out = Vec::new();
    if let Some(Json::Object(questions)) = body.get("questions") {
        let reversed: Vec<(String, Json)> = questions.iter().map(|(id, q)| {
            let q = match (q.get("type").and_then(Json::as_str), q.get("criteria")) {
                (Some("choice"), Some(Json::Object(kv))) => {
                    let mut kv = kv.clone();
                    kv.reverse();
                    set(q, "criteria", Json::Object(kv))
                }
                _ => q.clone(),
            };
            (id.clone(), q)
        }).collect();
        if reversed.iter().zip(questions).any(|(a, b)| a.1.to_python(false) != b.1.to_python(false)) {
            out.push(set(body, "questions", Json::Object(reversed)));
        }
    }
    if let Some(Json::Object(kv)) = body.get("state") {
        if kv.len() > 1 {
            let mut kv = kv.clone();
            rng.shuffle(&mut kv);
            out.push(set(body, "state", Json::Object(kv)));
        }
    }
    out
}

fn set(o: &Json, key: &str, value: Json) -> Json {
    match o {
        Json::Object(kv) => Json::Object(kv.iter().map(|(k, v)| (k.clone(), if k == key { value.clone() } else { v.clone() })).collect()),
        other => other.clone(),
    }
}

/// Request bodies read from a JSONL file, one per line: the family is the file's
/// stem. A line is a request body, or `{"body": ..., "gold": {question id: key}}`
/// when the answers are known, with a noul's as `true` / `false` and a score's as its
/// level index.
pub struct FileSource {
    family: String,
    bodies: Vec<(Json, Vec<Option<String>>)>,
}

impl FileSource {
    pub fn open(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut bodies = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row = Json::parse(line).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
            let (body, gold) = match (row.get("body"), row.get("gold")) {
                (Some(body), gold) => (body.clone(), gold),
                _ => (row.clone(), None),
            };
            let questions = body.get("questions").and_then(Json::as_object)
                .with_context(|| format!("{}: a request has questions", path.display()))?;
            let gold = questions.iter().map(|(id, _)| gold.and_then(|g| g.get(id)).and_then(|v| match v {
                Json::Str(s) => Some(s.clone()),
                Json::Bool(b) => Some(b.to_string()),
                Json::Int(d) => Some(d.clone()),
                _ => None,
            })).collect();
            bodies.push((body, gold));
        }
        let family = path.file_stem().and_then(|s| s.to_str()).unwrap_or("file").to_string();
        Ok(FileSource { family, bodies })
    }

    /// Every `*.jsonl` of `dir`, one source each, in name order.
    pub fn open_dir(dir: &Path) -> Result<Vec<Self>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect();
        paths.sort();
        paths.iter().map(|p| Self::open(p)).collect()
    }

    pub fn family(&self) -> &str { &self.family }

    pub fn len(&self) -> usize { self.bodies.len() }

    pub fn is_empty(&self) -> bool { self.bodies.is_empty() }

    pub fn draw(&self, rng: &mut Rng) -> Task {
        let (body, gold) = rng.pick(&self.bodies).clone();
        Task { family: self.family.clone(), body, gold }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_names_an_option_it_offers() {
        let mut rng = Rng::new(7);
        for family in FAMILIES {
            for _ in 0..40 {
                let t = generate(family, &mut rng);
                let questions = t.body.get("questions").and_then(Json::as_object).unwrap();
                assert_eq!(questions.len(), t.gold.len(), "{family}");
                for ((_, q), gold) in questions.iter().zip(&t.gold) {
                    let gold = gold.as_deref().expect("these families know their answers");
                    let ok = match q.get("type").and_then(Json::as_str).unwrap() {
                        "choice" => q.get("criteria").unwrap().get(gold).is_some(),
                        "noul" => gold == "true" || gold == "false",
                        "score" => gold.parse::<usize>().is_ok_and(|i| i < q.get("criteria").and_then(Json::as_array).unwrap().len()),
                        _ => false,
                    };
                    assert!(ok, "{family}: gold {gold} is not an option of {}", q.to_python(false));
                }
            }
        }
    }

    #[test]
    fn a_file_carries_its_answers_in_any_of_the_three_forms() {
        let dir = std::env::temp_dir().join(format!("ojas-pool-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mixed.jsonl");
        std::fs::write(&path, concat!(
            r#"{"body": {"state": "x", "questions": {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "score", "instructions": "?", "criteria": ["l", "h"]}}}, "gold": {"a": true, "b": 1}}"#, "\n",
            r#"{"state": "y", "questions": {"c": {"type": "choice", "instructions": "?", "criteria": {"k": null}}}}"#, "\n",
        )).unwrap();
        let src = FileSource::open(&path).unwrap();
        assert_eq!(src.family(), "mixed");
        assert_eq!(src.len(), 2);
        let mut rng = Rng::new(1);
        let golds: Vec<Vec<Option<String>>> = (0..20).map(|_| src.draw(&mut rng).gold).collect();
        assert!(golds.contains(&vec![Some("true".into()), Some("1".into())]));
        assert!(golds.contains(&vec![None]));
        assert_eq!(FileSource::open_dir(&dir).unwrap().len(), 1);
    }

    #[test]
    fn variants_keep_the_answers() {
        let mut rng = Rng::new(3);
        let t = generate("json_lookup", &mut rng);
        let vs = variants(&t.body, &mut rng);
        assert_eq!(vs.len(), 2, "a reversed choice and a shuffled state");
        for v in vs {
            let q = v.get("questions").unwrap().get("value").unwrap();
            assert!(q.get("criteria").unwrap().get(t.gold[0].as_deref().unwrap()).is_some());
            assert_eq!(v.get("state").unwrap().sorted().to_python(false), t.body.get("state").unwrap().sorted().to_python(false));
        }
    }
}
