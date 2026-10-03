//! The prompt a decision model was trained on, rendered from the Jinja template the
//! file ships (`tokenizer.chat_template.systemone`).
//!
//! The template receives the request's JSON as it arrived: a string is a string, and
//! anything else is a value the template prints with `tojson`. `tojson` writes what
//! Python's `json.dumps(value, ensure_ascii=False)` writes, keys in request order,
//! since that is how the training data serialized it.

use super::json::Json;
use anyhow::{Context, Result};
use minijinja::value::{Enumerator, Object, ObjectRepr, Value};
use minijinja::Environment;
use std::fmt;
use std::sync::Arc;

/// One option as the template sees it.
pub(crate) struct OptionView<'a> {
    pub(crate) key: &'a str,
    pub(crate) description: &'a Json,
    /// A label the model answers with, when the model's readout names one per
    /// option (`A`, `B`, ... `AA`).
    pub(crate) label: Option<&'a str>,
}

/// Everything a prompt is rendered from.
pub(crate) struct PromptInput<'a> {
    pub(crate) id: &'a str,
    pub(crate) kind: &'a str,
    pub(crate) instructions: &'a Json,
    pub(crate) state: &'a Json,
    pub(crate) options: Vec<OptionView<'a>>,
    /// One media marker per image.
    pub(crate) images: Vec<String>,
}

pub(crate) struct Template {
    env: Environment<'static>,
}

impl Template {
    pub(crate) fn new(source: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.add_template_owned("prompt", source.to_string()).context("parsing the decision template")?;
        Ok(Template { env })
    }

    pub(crate) fn render(&self, input: &PromptInput) -> Result<String> {
        let options: Vec<Value> = input.options.iter().map(|o| {
            let mut kv = vec![("key", Value::from(o.key)), ("description", value(o.description))];
            if let Some(label) = o.label { kv.push(("label", Value::from(label))); }
            Value::from_iter(kv)
        }).collect();
        let ctx = Value::from_iter([
            ("id", Value::from(input.id)),
            ("type", Value::from(input.kind)),
            ("instructions", value(input.instructions)),
            ("state", value(input.state)),
            ("options", Value::from(options)),
            ("images", Value::from(input.images.clone())),
        ]);
        self.env.get_template("prompt")?.render(ctx).context("rendering the decision template")
    }
}

/// A request value for the template: strings as strings, `null` as `none`, and
/// anything else as the JSON it is.
fn value(j: &Json) -> Value {
    match j {
        Json::Str(s) => Value::from(s.as_str()),
        Json::Null => Value::from(()),
        other => Value::from_object(JsonValue(other.clone())),
    }
}

/// A JSON number, boolean, array or object, keeping its exact text for `tojson`.
#[derive(Debug)]
struct JsonValue(Json);

impl Object for JsonValue {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        match self.0 {
            Json::Object(_) => ObjectRepr::Map,
            Json::Array(_) => ObjectRepr::Seq,
            _ => ObjectRepr::Plain,
        }
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        match &self.0 {
            Json::Object(kv) => {
                let k = key.as_str()?;
                kv.iter().find(|(name, _)| name == k).map(|(_, v)| value(v))
            }
            Json::Array(items) => items.get(key.as_usize()?).map(value),
            _ => None,
        }
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        match &self.0 {
            Json::Object(kv) => Enumerator::Values(kv.iter().map(|(k, _)| Value::from(k.as_str())).collect()),
            Json::Array(items) => Enumerator::Seq(items.len()),
            _ => Enumerator::NonEnumerable,
        }
    }

    /// Python's truthiness, which the templates test with `{% if %}`.
    fn is_true(self: &Arc<Self>) -> bool {
        match &self.0 {
            Json::Bool(b) => *b,
            Json::Int(i) => !i.trim_start_matches(['-', '0']).is_empty(),
            Json::Float(f) => *f != 0.0,
            Json::Array(a) => !a.is_empty(),
            Json::Object(o) => !o.is_empty(),
            Json::Null | Json::Str(_) => unreachable!("strings and null are not wrapped"),
        }
    }

    fn render(self: &Arc<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.to_python(false))
    }
}

/// `json.dumps(value, ensure_ascii=False)`.
fn tojson(v: Value) -> String {
    if let Some(j) = v.downcast_object_ref::<JsonValue>() { return j.0.to_python(false); }
    if let Some(s) = v.as_str() { return Json::Str(s.to_string()).to_python(false); }
    if v.is_none() || v.is_undefined() { return "null".into(); }
    v.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(src: &str, state: &str, instructions: &str, options: &[(&str, &str)]) -> String {
        let (state, instructions) = (Json::parse(state).unwrap(), Json::parse(instructions).unwrap());
        let descs: Vec<Json> = options.iter().map(|(_, d)| Json::parse(d).unwrap()).collect();
        let input = PromptInput {
            id: "q", kind: "choice", instructions: &instructions, state: &state,
            options: options.iter().zip(&descs).map(|((k, _), d)| OptionView { key: k, description: d, label: None }).collect(),
            images: Vec::new(),
        };
        Template::new(src).unwrap().render(&input).unwrap()
    }

    #[test]
    fn json_values_print_as_python_dumps_them_and_strings_print_raw() {
        let src = "{{ state if state is string else state | tojson }}|{{ instructions if instructions is string else instructions | tojson }}";
        assert_eq!(render(src, r#"{"b": [1, 2.5, true, null], "a": "é\n"}"#, r#""raw \"text\"""#, &[]),
            "{\"b\": [1, 2.5, true, null], \"a\": \"é\\n\"}|raw \"text\"");
    }

    #[test]
    fn options_loops_letters_and_truthiness_work_as_the_templates_use_them() {
        let src = "{% set letters = 'ABC' %}{% for o in options %}[{{ letters[loop.index0] }}] {{ o.key }}\
                   {% if o.description %}: {{ o.description if o.description is string else o.description | tojson }}{% endif %} \
                   {% endfor %}{{ options | length - 1 }}";
        assert_eq!(render(src, "null", "null", &[("x", "null"), ("y", r#""why""#), ("z", "{}"), ("w", "[0]")]),
            "[A] x [B] y: why [C] z [] w: [0] 3");
    }
}
