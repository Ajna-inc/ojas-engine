//! JSON Schema to GBNF.
//!
//! Supported: `type` (one or a list), object `properties` / `required` /
//! `additionalProperties`, `enum`, `const`, `anyOf` / `oneOf`, `allOf` over object
//! schemas, `$ref` into `$defs` or `definitions` (recursion included), array
//! `items` / `prefixItems` / `minItems` / `maxItems`, string `minLength` /
//! `maxLength` and the formats `date`, `time`, `date-time` and `uuid`.
//!
//! Properties come out in the order the schema declares them, required ones
//! always and optional ones possibly. An object accepts properties beyond its
//! declared ones only when `additionalProperties` is `true` or a schema: for
//! generation, an unlisted key is far more often a mistake than intent. `pattern`
//! and numeric bounds are not expressible in the grammar and are not enforced.
//! Anything else unrecognised is an error rather than a silently looser grammar.

use crate::gbnf::Grammar;
use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Shared building blocks. `space` bounds whitespace so a model cannot fill its
/// budget with indentation.
const PRIMITIVES: &str = r#"
space ::= | " " | "\n" [ \t]{0,20}
boolean ::= ( "true" | "false" ) space
null ::= "null" space
char ::= [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} )
string ::= "\"" char* "\"" space
integral-part ::= "0" | [1-9] [0-9]{0,15}
decimal-part ::= [0-9]{1,16}
integer ::= "-"? integral-part space
number ::= "-"? integral-part ( "." decimal-part )? ( [eE] [-+]? integral-part )? space
value ::= object | array | string | number | boolean | null
object ::= "{" space ( string ":" space value ( "," space string ":" space value )* )? "}" space
array ::= "[" space ( value ( "," space value )* )? "]" space
"#;

const FORMATS: &str = r#"
date-body ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [12] [0-9] | "3" [01] )
time-body ::= ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{1,9} )? ( "Z" | [+-] ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
date-string ::= "\"" date-body "\"" space
time-string ::= "\"" time-body "\"" space
date-time-string ::= "\"" date-body "T" time-body "\"" space
uuid-string ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space
"#;

impl Grammar {
    /// Any JSON object: OpenAI's `response_format: {"type": "json_object"}`.
    pub fn json() -> Grammar {
        Grammar::parse(&format!("root ::= object\n{PRIMITIVES}")).expect("the built-in JSON grammar parses")
    }

    /// A grammar for exactly the JSON documents `schema` describes.
    pub fn from_json_schema(schema: &Value) -> Result<Grammar> {
        Grammar::parse(&schema_to_gbnf(schema)?)
    }
}

/// The GBNF text for `schema`, rooted at `root`.
pub fn schema_to_gbnf(schema: &Value) -> Result<String> {
    let mut c = Compiler { root: schema, rules: Vec::new(), names: HashMap::new(), refs: HashMap::new() };
    let top = c.visit(schema, "root-value")?;
    let mut out = format!("root ::= {top}\n");
    for (name, body) in &c.rules { out.push_str(&format!("{name} ::= {body}\n")); }
    out.push_str(PRIMITIVES);
    out.push_str(FORMATS);
    Ok(out)
}

struct Compiler<'a> {
    root: &'a Value,
    rules: Vec<(String, String)>,
    names: HashMap<String, usize>,
    refs: HashMap<String, String>,
}

/// A GBNF string literal matching `text` exactly.
fn literal(text: &str) -> String {
    let mut s = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\x{:02X}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
    s
}

/// A JSON value as it must appear in the output.
fn json_literal(v: &Value) -> String { format!("{} space", literal(&v.to_string())) }

impl Compiler<'_> {
    /// Add a rule named after `hint` and return its name.
    fn rule(&mut self, hint: &str, body: String) -> String {
        let base: String = hint.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
        let base = if base.is_empty() { "r".to_string() } else { base };
        let n = self.names.entry(base.clone()).or_insert(0);
        *n += 1;
        let name = if *n == 1 { base } else { format!("{base}-{n}") };
        self.rules.push((name.clone(), body));
        name
    }

    fn resolve(&self, r: &str) -> Result<&Value> {
        let path = r.strip_prefix("#/").with_context(|| format!("only local `$ref`s are supported, not `{r}`"))?;
        let mut v = self.root;
        for part in path.split('/') {
            let key = part.replace("~1", "/").replace("~0", "~");
            v = v.get(&key).with_context(|| format!("`$ref` `{r}` does not resolve"))?;
        }
        Ok(v)
    }

    /// A GBNF expression (usually a rule name) matching `schema`.
    fn visit(&mut self, schema: &Value, hint: &str) -> Result<String> {
        let obj = match schema {
            Value::Bool(true) => return Ok("value".into()),
            Value::Bool(false) => bail!("schema `false` matches nothing"),
            Value::Object(o) => o,
            _ => bail!("a schema must be an object or a boolean"),
        };
        if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
            if let Some(name) = self.refs.get(r) { return Ok(name.clone()); }
            // Reserve the name first so a recursive reference resolves to it.
            let hint = r.rsplit('/').next().unwrap_or("ref").to_string();
            let name = self.rule(&hint, String::new());
            self.refs.insert(r.to_string(), name.clone());
            let target = self.resolve(r)?.clone();
            let body = self.visit(&target, &format!("{hint}-body"))?;
            let slot = self.rules.iter_mut().find(|(n, _)| *n == name).unwrap();
            slot.1 = body;
            return Ok(name);
        }
        if let Some(c) = obj.get("const") { return Ok(self.rule(hint, json_literal(c))); }
        if let Some(e) = obj.get("enum") {
            let vals = e.as_array().context("`enum` must be an array")?;
            if vals.is_empty() { bail!("`enum` is empty"); }
            let alts: Vec<String> = vals.iter().map(json_literal).collect();
            return Ok(self.rule(hint, alts.join(" | ")));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(list) = obj.get(key) {
                let list = list.as_array().with_context(|| format!("`{key}` must be an array"))?;
                let mut alts = Vec::new();
                for (i, s) in list.iter().enumerate() { alts.push(self.visit(s, &format!("{hint}-{i}"))?); }
                return Ok(self.rule(hint, alts.join(" | ")));
            }
        }
        if let Some(list) = obj.get("allOf") {
            let merged = self.merge_all_of(list)?;
            return self.visit(&merged, hint);
        }
        match obj.get("type") {
            Some(Value::Array(types)) => {
                let mut alts = Vec::new();
                for t in types {
                    let mut one = obj.clone();
                    one.insert("type".into(), t.clone());
                    alts.push(self.visit(&Value::Object(one), &format!("{hint}-{}", t.as_str().unwrap_or("t")))?);
                }
                Ok(self.rule(hint, alts.join(" | ")))
            }
            Some(Value::String(t)) => self.typed(t, obj, hint),
            Some(_) => bail!("`type` must be a string or an array of strings"),
            None if obj.contains_key("properties") => self.typed("object", obj, hint),
            None if obj.contains_key("items") || obj.contains_key("prefixItems") => self.typed("array", obj, hint),
            None => Ok("value".into()),
        }
    }

    fn merge_all_of(&self, list: &Value) -> Result<Value> {
        let list = list.as_array().context("`allOf` must be an array")?;
        let mut props = Map::new();
        let mut required: Vec<Value> = Vec::new();
        for s in list {
            let s = match s.get("$ref").and_then(Value::as_str) { Some(r) => self.resolve(r)?, None => s };
            let o = s.as_object().context("`allOf` entries must be object schemas")?;
            if o.get("type").is_some_and(|t| t != "object") || !o.contains_key("properties") {
                bail!("`allOf` is supported only over object schemas with `properties`");
            }
            if let Some(p) = o.get("properties").and_then(Value::as_object) {
                for (k, v) in p { props.insert(k.clone(), v.clone()); }
            }
            if let Some(r) = o.get("required").and_then(Value::as_array) { required.extend(r.iter().cloned()); }
        }
        Ok(serde_json::json!({"type": "object", "properties": props, "required": required}))
    }

    fn typed(&mut self, t: &str, obj: &Map<String, Value>, hint: &str) -> Result<String> {
        let u = |k: &str| obj.get(k).and_then(Value::as_u64);
        Ok(match t {
            "null" => "null".into(),
            "boolean" => "boolean".into(),
            "integer" => "integer".into(),
            "number" => "number".into(),
            "string" => match obj.get("format").and_then(Value::as_str) {
                Some("date") => "date-string".into(),
                Some("time") => "time-string".into(),
                Some("date-time") => "date-time-string".into(),
                Some("uuid") => "uuid-string".into(),
                _ => match (u("minLength"), u("maxLength")) {
                    (None, None) => "string".into(),
                    (lo, hi) => {
                        let lo = lo.unwrap_or(0);
                        let rep = match hi { Some(h) => format!("{{{lo},{h}}}"), None => format!("{{{lo},}}") };
                        self.rule(hint, format!("\"\\\"\" char{rep} \"\\\"\" space"))
                    }
                },
            },
            "array" => self.array(obj, hint)?,
            "object" => self.object(obj, hint)?,
            other => bail!("unknown type `{other}`"),
        })
    }

    fn array(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<String> {
        if let Some(prefix) = obj.get("prefixItems").and_then(Value::as_array) {
            let mut parts = Vec::new();
            for (i, s) in prefix.iter().enumerate() { parts.push(self.visit(s, &format!("{hint}-{i}"))?); }
            let body = if parts.is_empty() { String::new() } else { parts.join(" \",\" space ") };
            return Ok(self.rule(hint, format!("\"[\" space {body} \"]\" space")));
        }
        let item = match obj.get("items") { Some(s) => self.visit(s, &format!("{hint}-item"))?, None => "value".into() };
        let lo = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        let hi = obj.get("maxItems").and_then(Value::as_u64);
        if hi.is_some_and(|h| h < lo) { bail!("`maxItems` is below `minItems`"); }
        let more = |n_lo: u64, n_hi: Option<u64>| -> String {
            match n_hi {
                Some(h) => format!("( \",\" space {item} ){{{n_lo},{h}}}"),
                None if n_lo == 0 => format!("( \",\" space {item} )*"),
                None => format!("( \",\" space {item} ){{{n_lo},}}"),
            }
        };
        let body = match (lo, hi) {
            (_, Some(0)) => String::new(),
            (0, h) => format!("( {item} {} )?", more(0, h.map(|h| h - 1))),
            (l, h) => format!("{item} {}", more(l - 1, h.map(|h| h - 1))),
        };
        Ok(self.rule(hint, format!("\"[\" space {body} \"]\" space")))
    }

    fn object(&mut self, obj: &Map<String, Value>, hint: &str) -> Result<String> {
        let required: Vec<&str> = obj.get("required").and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let mut props: Vec<(String, bool)> = Vec::new();
        if let Some(p) = obj.get("properties").and_then(Value::as_object) {
            for (key, schema) in p {
                let v = self.visit(schema, &format!("{hint}-{key}"))?;
                let kv = self.rule(&format!("{hint}-{key}-kv"), format!("{} space \":\" space {v}", literal(&Value::String(key.clone()).to_string())));
                props.push((kv, required.contains(&key.as_str())));
            }
        }
        for r in &required {
            if !obj.get("properties").and_then(Value::as_object).is_some_and(|p| p.contains_key(*r)) {
                bail!("required property `{r}` is not declared in `properties`");
            }
        }
        let extra = match obj.get("additionalProperties") {
            Some(Value::Bool(true)) => Some("value".to_string()),
            Some(s @ Value::Object(_)) => Some(self.visit(s, &format!("{hint}-additional"))?),
            _ => None,
        };
        let extra_kv = extra.map(|v| self.rule(&format!("{hint}-extra-kv"), format!("string \":\" space {v}")));
        // tail(i, comma): the properties from i on, each preceded by a comma when
        // `comma` (something was already written). Optional properties branch, so
        // each (i, comma) pair gets one rule rather than an exponential expansion.
        let mut memo: HashMap<(usize, bool), String> = HashMap::new();
        let body = self.tail(&props, 0, false, extra_kv.as_deref(), hint, &mut memo);
        Ok(self.rule(hint, format!("\"{{\" space {body} \"}}\" space")))
    }

    fn tail(&mut self, props: &[(String, bool)], i: usize, comma: bool, extra: Option<&str>,
            hint: &str, memo: &mut HashMap<(usize, bool), String>) -> String {
        if let Some(r) = memo.get(&(i, comma)) { return r.clone(); }
        let sep = if comma { "\",\" space " } else { "" };
        let expr = if i == props.len() {
            match (extra, comma) {
                (None, _) => String::new(),
                (Some(kv), true) => format!("( \",\" space {kv} )*"),
                (Some(kv), false) => format!("( {kv} ( \",\" space {kv} )* )?"),
            }
        } else {
            let (kv, req) = &props[i];
            let with = format!("{sep}{kv} {}", self.tail(props, i + 1, true, extra, hint, memo));
            if *req { with } else { format!("( {with} | {} )", self.tail(props, i + 1, comma, extra, hint, memo)) }
        };
        let name = self.rule(&format!("{hint}-tail-{i}"), if expr.is_empty() { "\"\"".into() } else { expr });
        memo.insert((i, comma), name.clone());
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn g(schema: Value) -> Grammar { Grammar::from_json_schema(&schema).unwrap() }

    #[test]
    fn generic_json_object() {
        let gr = Grammar::json();
        assert!(gr.accepts(r#"{"a": [1, 2.5, -3e2, true, null, "x\"y"], "b": {}}"#));
        assert!(!gr.accepts(r#"[1]"#) && !gr.accepts(r#"{"a" 1}"#) && !gr.accepts(r#"{"a": 01}"#));
    }

    #[test]
    fn required_and_optional_properties_in_order() {
        let gr = g(json!({"type": "object", "properties": {
            "name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}
        }, "required": ["name"]}));
        for ok in [r#"{"name": "a"}"#, r#"{"name": "a", "age": 3}"#, r#"{"name": "a", "tags": ["x"]}"#, r#"{"name":"a","age":3,"tags":[]}"#] {
            assert!(gr.accepts(ok), "{ok}");
        }
        for bad in [r#"{}"#, r#"{"age": 3}"#, r#"{"age": 3, "name": "a"}"#, r#"{"name": "a", "other": 1}"#, r#"{"name": 1}"#] {
            assert!(!gr.accepts(bad), "{bad}");
        }
    }

    #[test]
    fn all_optional_object() {
        let gr = g(json!({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}));
        for ok in ["{}", r#"{"a": 1}"#, r#"{"b": 2}"#, r#"{"a": 1, "b": 2}"#] { assert!(gr.accepts(ok), "{ok}"); }
        assert!(!gr.accepts(r#"{, "b": 2}"#) && !gr.accepts(r#"{"a": 1,}"#));
    }

    #[test]
    fn additional_properties_when_allowed() {
        let gr = g(json!({"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": {"type": "string"}}));
        assert!(gr.accepts(r#"{"a": 1, "x": "y", "z": "w"}"#) && gr.accepts(r#"{"x": "y"}"#));
        assert!(!gr.accepts(r#"{"a": 1, "x": 2}"#));
    }

    #[test]
    fn enum_const_any_of_and_nullable() {
        let gr = g(json!({"type": "object", "properties": {
            "color": {"enum": ["red", "green"]}, "v": {"const": 1},
            "x": {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "n": {"type": ["string", "null"]}
        }, "required": ["color", "v", "x", "n"]}));
        assert!(gr.accepts(r#"{"color": "red", "v": 1, "x": "s", "n": null}"#));
        assert!(gr.accepts(r#"{"color": "green", "v": 1, "x": 5, "n": "t"}"#));
        assert!(!gr.accepts(r#"{"color": "blue", "v": 1, "x": 5, "n": null}"#));
        assert!(!gr.accepts(r#"{"color": "red", "v": 2, "x": 5, "n": null}"#));
    }

    #[test]
    fn arrays_with_bounds_and_tuples() {
        let gr = g(json!({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}));
        assert!(!gr.accepts("[]") && gr.accepts("[1]") && gr.accepts("[1, 2, 3]") && !gr.accepts("[1, 2, 3, 4]"));
        let t = g(json!({"type": "array", "prefixItems": [{"type": "string"}, {"type": "boolean"}]}));
        assert!(t.accepts(r#"["a", true]"#) && !t.accepts(r#"["a"]"#));
    }

    #[test]
    fn recursive_refs_and_formats() {
        let gr = g(json!({
            "$defs": {"node": {"type": "object", "properties": {
                "v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}
            }, "required": ["v"]}},
            "$ref": "#/$defs/node"
        }));
        assert!(gr.accepts(r#"{"v": 1, "kids": [{"v": 2}, {"v": 3, "kids": []}]}"#));
        let d = g(json!({"type": "string", "format": "date"}));
        assert!(d.accepts(r#""2026-10-01""#) && !d.accepts(r#""2026-13-01""#));
    }

    #[test]
    fn string_length_bounds() {
        let gr = g(json!({"type": "string", "minLength": 2, "maxLength": 3}));
        assert!(!gr.accepts(r#""a""#) && gr.accepts(r#""ab""#) && gr.accepts(r#""abc""#) && !gr.accepts(r#""abcd""#));
    }

    #[test]
    fn unsupported_constructs_are_errors() {
        assert!(Grammar::from_json_schema(&json!({"type": "tuple"})).is_err());
        assert!(Grammar::from_json_schema(&json!({"$ref": "https://example.com/s"})).is_err());
        assert!(Grammar::from_json_schema(&json!({"type": "object", "properties": {}, "required": ["x"]})).is_err());
    }
}
