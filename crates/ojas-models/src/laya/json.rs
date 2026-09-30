//! Order-preserving JSON, and Python's `json.dumps` text form.
//!
//! Laya's reference implementation turns a JSON state into text with
//! `json.dumps(state, ensure_ascii=False)` before tokenizing it, and reads a
//! `choice` question's options in the order its object lists them. Both orders
//! reach the model: key order changes the state's tokens, option order changes
//! which `[MASK]` scores which option. So objects here keep their keys in source
//! order, and [`Json::to_python`] reproduces `json.dumps` byte for byte: `", "`
//! and `": "` separators, Python's string escapes, `repr` for floats.

use anyhow::{bail, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal, kept as its decimal digits (Python ints are unbounded).
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Keys in source order. A repeated key keeps its first position and its last
    /// value, as `json.loads` builds a dict.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.s.len() { bail!("JSON: trailing characters at byte {}", p.i); }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        if let Json::Array(v) = self { Some(v) } else { None }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        if let Json::Object(kv) = self { Some(kv) } else { None }
    }

    /// A number as f64 (integers included).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Float(f) => Some(*f),
            Json::Int(digits) => digits.parse().ok(),
            _ => None,
        }
    }

    /// `json.dumps(self, ensure_ascii=...)`.
    pub fn to_python(&self, ensure_ascii: bool) -> String {
        let mut out = String::new();
        self.write_python(&mut out, ensure_ascii);
        out
    }

    fn write_python(&self, out: &mut String, ascii: bool) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(digits) => out.push_str(digits),
            Json::Float(f) => out.push_str(&python_float_repr(*f)),
            Json::Str(s) => write_python_str(out, s, ascii),
            Json::Array(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 { out.push_str(", "); }
                    v.write_python(out, ascii);
                }
                out.push(']');
            }
            Json::Object(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 { out.push_str(", "); }
                    write_python_str(out, k, ascii);
                    out.push_str(": ");
                    v.write_python(out, ascii);
                }
                out.push('}');
            }
        }
    }
}

/// `json.encoder`'s string form: `"` and `\` escaped, the five short control
/// escapes, other characters below U+0020 as `\u00XX`. With `ascii`, every
/// non-ASCII character is `\uXXXX` too, astral ones as a surrogate pair.
fn write_python_str(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && !c.is_ascii() => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) { out.push_str(&format!("\\u{u:04x}")); }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that round-trip, in fixed notation
/// when the decimal exponent is in `[-4, 16)` and scientific otherwise, with a
/// trailing `.0` on integral fixed values (`1.0`, `1e+16`, `1.5e-05`).
fn python_float_repr(x: f64) -> String {
    if x.is_nan() { return "NaN".into(); }
    if x.is_infinite() { return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }; }
    if x == 0.0 { return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() }; }
    // Rust's `{:e}` is the shortest round-trip form: "d.ddde<exp>".
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp < 0 {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        } else if exp + 1 >= n {
            format!("{}{}.0", digits, "0".repeat((exp + 1 - n) as usize))
        } else {
            let (a, b) = digits.split_at((exp + 1) as usize);
            format!("{a}.{b}")
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() == 1 { digits } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

struct Parser<'s> {
    s: &'s [u8],
    i: usize,
}

/// Nesting deeper than this is refused rather than recursed into.
const MAX_DEPTH: usize = 256;

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') { self.i += 1; }
    }

    fn peek(&self) -> Option<u8> { self.s.get(self.i).copied() }

    fn expect(&mut self, lit: &str) -> Result<()> {
        if self.s[self.i..].starts_with(lit.as_bytes()) { self.i += lit.len(); Ok(()) }
        else { bail!("JSON: expected `{lit}` at byte {}", self.i) }
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > MAX_DEPTH { bail!("JSON: nesting deeper than {MAX_DEPTH}"); }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => { self.expect("true")?; Ok(Json::Bool(true)) }
            Some(b'f') => { self.expect("false")?; Ok(Json::Bool(false)) }
            Some(b'n') => { self.expect("null")?; Ok(Json::Null) }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(c) => bail!("JSON: unexpected `{}` at byte {}", c as char, self.i),
            None => bail!("JSON: unexpected end of input"),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json> {
        self.i += 1;
        let mut kv: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') { self.i += 1; return Ok(Json::Object(kv)); }
        loop {
            self.ws();
            if self.peek() != Some(b'"') { bail!("JSON: expected a key at byte {}", self.i); }
            let k = self.string()?;
            self.ws();
            self.expect(":")?;
            self.ws();
            let v = self.value(depth + 1)?;
            match kv.iter_mut().find(|(ek, _)| *ek == k) {
                Some(slot) => slot.1 = v,
                None => kv.push((k, v)),
            }
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => { self.i += 1; return Ok(Json::Object(kv)); }
                _ => bail!("JSON: expected `,` or `}}` at byte {}", self.i),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') { self.i += 1; return Ok(Json::Array(items)); }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => { self.i += 1; return Ok(Json::Array(items)); }
                _ => bail!("JSON: expected `,` or `]` at byte {}", self.i),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| anyhow::anyhow!("JSON: truncated \\u escape"))?;
        let v = u32::from_str_radix(std::str::from_utf8(h)?, 16)
            .map_err(|_| anyhow::anyhow!("JSON: bad \\u escape at byte {}", self.i))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                if self.s[self.i] < 0x20 { bail!("JSON: raw control character in string at byte {}", self.i); }
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[start..self.i])?);
            match self.peek() {
                Some(b'"') => { self.i += 1; return Ok(out); }
                Some(b'\\') => {
                    self.i += 1;
                    let e = self.peek().ok_or_else(|| anyhow::anyhow!("JSON: truncated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) && self.s[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) { bail!("JSON: unpaired surrogate"); }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else { hi };
                            out.push(char::from_u32(cp).ok_or_else(|| anyhow::anyhow!("JSON: unpaired surrogate"))?);
                        }
                        c => bail!("JSON: bad escape `\\{}`", c as char),
                    }
                }
                _ => bail!("JSON: unterminated string"),
            }
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        if self.peek() == Some(b'-') { self.i += 1; }
        let digits = |p: &mut Self| { let s = p.i; while p.peek().is_some_and(|c| c.is_ascii_digit()) { p.i += 1; } p.i - s };
        let int_digits = digits(self);
        if int_digits == 0 { bail!("JSON: bad number at byte {start}"); }
        if int_digits > 1 && self.s[self.i - int_digits] == b'0' { bail!("JSON: leading zero at byte {start}"); }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            self.i += 1;
            if digits(self) == 0 { bail!("JSON: bad fraction at byte {start}"); }
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) { self.i += 1; }
            if digits(self) == 0 { bail!("JSON: bad exponent at byte {start}"); }
            is_float = true;
        }
        let text = std::str::from_utf8(&self.s[start..self.i])?;
        if is_float {
            Ok(Json::Float(text.parse()?))
        } else if text.trim_start_matches('-').bytes().all(|c| c == b'0') {
            Ok(Json::Int("0".into()))
        } else {
            Ok(Json::Int(text.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    /// Expected strings are Python 3's `json.dumps(json.loads(src), ensure_ascii=...)`.
    #[test]
    fn matches_python_json_dumps() {
        let cases: &[(&str, bool, &str)] = &[
            (r#"{"from":"user@acme.com","subject":"Invoice #4411","n":3}"#, false,
             r#"{"from": "user@acme.com", "subject": "Invoice #4411", "n": 3}"#),
            (r#"{"b":1,"a":[1,2.5,{"z":null,"y":true}]}"#, false,
             r#"{"b": 1, "a": [1, 2.5, {"z": null, "y": true}]}"#),
            (r#"{"k":1,"j":2,"k":3}"#, false, r#"{"k": 3, "j": 2}"#),
            (r#"["café","日本","😀","tab\there","q\"uote","back\\slash","\u0001"]"#, false,
             "[\"café\", \"日本\", \"😀\", \"tab\\there\", \"q\\\"uote\", \"back\\\\slash\", \"\\u0001\"]"),
            (r#"["café","😀"]"#, true, r#"["caf\u00e9", "\ud83d\ude00"]"#),
            ("[1.0, 1e2, 1.5e-5, 0.0001, 1e16, 123456789012345678, -0, -0.0, 0.1, 1e-7, 2.5E+20, 12345.678]", false,
             "[1.0, 100.0, 1.5e-05, 0.0001, 1e+16, 123456789012345678, 0, -0.0, 0.1, 1e-07, 2.5e+20, 12345.678]"),
            ("{}", false, "{}"),
            ("[]", false, "[]"),
            ("  \"plain\"  ", false, "\"plain\""),
        ];
        for &(src, ascii, want) in cases {
            let got = Json::parse(src).unwrap_or_else(|e| panic!("{src}: {e}")).to_python(ascii);
            assert_eq!(got, want, "json.dumps({src}, ensure_ascii={ascii})");
        }
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in ["{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "tru", "[1] 2", "\"a\nb\""] {
            assert!(Json::parse(bad).is_err(), "accepted {bad:?}");
        }
    }
}
