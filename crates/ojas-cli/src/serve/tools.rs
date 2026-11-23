//! OpenAI tool calls on `/v1/chat/completions`: `tools` in, `tool_calls` out.
//!
//! The request is rewritten into the Messages API's form and rendered by the same
//! prompt builder (`anthropic::convert`), so both routes declare tools and read
//! calls back one way. Text before a call streams as it is generated; the calls
//! themselves go out whole once generation ends, since a call is only known to be
//! one when it is complete.

use super::anthropic::{self, Parsed, Shape};
use serde_json::{json, Map, Value};

/// What opens a call in the generated text.
const CALL: &str = "<tool_call>";

/// How a reply with tools is read back.
pub(crate) struct ToolReply {
    shape: Shape,
    /// Text the prompt already ends with when the request forces a call; the
    /// generated text continues it.
    prefill: String,
    /// Bytes of the reply's visible text already streamed.
    shown: usize,
}

enum Choice {
    Auto,
    None,
    Required,
    Named(String),
}

fn choice(body: &Value) -> Result<Choice, String> {
    match body.get("tool_choice") {
        None | Some(Value::Null) => Ok(Choice::Auto),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => Ok(Choice::Auto),
            "none" => Ok(Choice::None),
            "required" => Ok(Choice::Required),
            other => Err(format!("unsupported tool_choice \"{other}\"")),
        },
        Some(v) => v.pointer("/function/name").and_then(Value::as_str).map(|n| Choice::Named(n.to_string()))
            .ok_or_else(|| "tool_choice must be \"auto\", \"none\", \"required\" or {\"type\": \"function\", \"function\": {\"name\"}}".to_string()),
    }
}

/// A call's arguments as an object: OpenAI sends them as a JSON string.
fn arguments(v: Option<&Value>) -> Value {
    match v {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(o @ Value::Object(_)) => o.clone(),
        _ => json!({}),
    }
}

/// The request in the server's internal form (a rendered `prompt` beside the
/// request's other fields) when it declares tools or carries a tool exchange;
/// `None` for a plain chat, which keeps the model's ordinary template.
pub(crate) fn convert(body: &Value, arch: &str, default_system: &str) -> Result<Option<(Value, ToolReply)>, String> {
    let Some(msgs) = body.get("messages").and_then(Value::as_array) else { return Ok(None) };
    let declared: &[Value] = body.get("tools").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
    let history = msgs.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("tool")
        || m.get("tool_calls").and_then(Value::as_array).is_some_and(|c| !c.is_empty()));
    let choice = choice(body)?;
    let offered = !declared.is_empty() && !matches!(choice, Choice::None);
    if !offered && !history { return Ok(None); }

    // The prompt builder writes the ChatML tool format; a model with another chat
    // format would read it as plain text.
    if ojas_tokenize::chat_eos(arch) != "<|im_end|>" {
        return Err(format!("tool calls are not supported for the chat format of architecture \"{arch}\""));
    }
    if super::images::has_images(body) {
        return Err("a request with tools cannot carry images".to_string());
    }
    if offered && ["response_format", "grammar", "json_schema"].iter().any(|k| body.get(k).is_some_and(|v| !v.is_null())) {
        return Err("give either tools or an output format, not both: a constrained reply cannot hold a call".to_string());
    }

    let mut tools = Vec::new();
    for t in declared {
        let f = t.get("function").unwrap_or(t);
        let name = f.get("name").and_then(Value::as_str).filter(|n| !n.is_empty())
            .ok_or("every tool needs function.name")?;
        tools.push(json!({
            "name": name,
            "description": f.get("description").cloned().unwrap_or(json!("")),
            "input_schema": f.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        }));
    }
    let prefill = match &choice {
        Choice::Required if offered => format!("{CALL}\n<function="),
        Choice::Named(name) => {
            if !tools.iter().any(|t| t["name"] == name.as_str()) {
                return Err(format!("tool_choice names \"{name}\", which is not among the tools"));
            }
            format!("{CALL}\n<function={name}>\n")
        }
        _ => String::new(),
    };

    let mut system: Vec<String> = Vec::new();
    let mut turns: Vec<Value> = Vec::new();
    for m in msgs {
        let text = super::images::content_text(m.get("content"));
        match m.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => system.push(text),
            "assistant" => {
                let mut blocks = Vec::new();
                if !text.trim().is_empty() { blocks.push(json!({"type": "text", "text": text})); }
                for c in m.get("tool_calls").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default() {
                    blocks.push(json!({
                        "type": "tool_use", "id": c.get("id").cloned().unwrap_or(Value::Null),
                        "name": c.pointer("/function/name").cloned().unwrap_or(json!("")),
                        "input": arguments(c.pointer("/function/arguments")),
                    }));
                }
                turns.push(json!({"role": "assistant", "content": blocks}));
            }
            "tool" => {
                let result = json!({"type": "tool_result", "tool_use_id": m.get("tool_call_id").cloned().unwrap_or(Value::Null),
                                    "content": text});
                // The results of one assistant turn's calls share one user turn.
                match turns.last_mut().filter(|t| t["role"] == "user" && t["content"].as_array()
                    .is_some_and(|b| b.iter().all(|x| x["type"] == "tool_result"))) {
                    Some(turn) => turn["content"].as_array_mut().unwrap().push(result),
                    None => turns.push(json!({"role": "user", "content": [result]})),
                }
            }
            _ => turns.push(json!({"role": "user", "content": text})),
        }
    }
    let mut request = json!({"messages": turns, "tools": if offered { tools } else { Vec::new() }});
    if !system.is_empty() { request["system"] = json!(system.join("\n\n")); }
    let rendered = anthropic::convert(&request, default_system)?;

    let mut internal: Map<String, Value> = body.as_object().cloned().unwrap_or_default();
    for k in ["messages", "tools", "tool_choice", "parallel_tool_calls"] { internal.remove(k); }
    let prompt = rendered.body["prompt"].as_str().unwrap_or_default();
    internal.insert("prompt".into(), json!(format!("{prompt}{prefill}")));
    Ok(Some((Value::Object(internal), ToolReply { shape: rendered.shape, prefill, shown: 0 })))
}

/// The part of `text` a client may see so far, as a byte range: nothing from a
/// call's opening tag on, no tail that may be the start of one, and no whitespace
/// around the rest.
fn visible(text: &str) -> (usize, usize) {
    let end = text.find(CALL).unwrap_or_else(|| {
        (1..CALL.len()).rev().find(|&k| text.ends_with(&CALL[..k])).map_or(text.len(), |k| text.len() - k)
    });
    let start = (text.len() - text.trim_start().len()).min(end);
    (start, text[..end].trim_end().len().max(start))
}

impl ToolReply {
    /// The visible text not yet streamed, given everything generated so far.
    pub(crate) fn delta<'t>(&mut self, text: &'t str) -> &'t str {
        // A forced call starts inside the call: none of the text is visible.
        if !self.prefill.is_empty() { return ""; }
        let (start, end) = visible(text);
        let from = (start + self.shown).min(end);
        self.shown = end - start;
        &text[from..end]
    }

    /// The whole reply once generation ended, and the visible text still unsent.
    pub(crate) fn finish(&self, text: &str) -> (Parsed, String) {
        let parsed = self.shape.parse(&format!("{}{text}", self.prefill));
        let rest = parsed.text.get(self.shown..).unwrap_or_default().to_string();
        (parsed, rest)
    }
}

/// `tool_calls` for a message, or with `index` for a stream's delta.
pub(crate) fn calls_json(calls: &[(String, Value)], id: &str, indexed: bool) -> Vec<Value> {
    calls.iter().enumerate().map(|(k, (name, input))| {
        let mut call = json!({"id": format!("call_{id}{k}"), "type": "function",
                              "function": {"name": name, "arguments": input.to_string()}});
        if indexed { call["index"] = json!(k); }
        call
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "description": "weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}])
    }

    #[test]
    fn a_plain_chat_is_left_to_the_models_own_template() {
        assert!(convert(&json!({"messages": [{"role": "user", "content": "hi"}]}), "qwen3", "").unwrap().is_none());
        let off = json!({"messages": [{"role": "user", "content": "hi"}], "tools": weather(), "tool_choice": "none"});
        assert!(convert(&off, "qwen3", "").unwrap().is_none());
    }

    #[test]
    fn a_tool_exchange_renders_with_its_calls_and_results() {
        let body = json!({
            "tools": weather(), "temperature": 0.2, "max_tokens": 64,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "weather in Pune?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Pune\", \"days\": 2}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "31C"},
                {"role": "tool", "tool_call_id": "call_2", "content": "dry"},
            ]});
        let (internal, _) = convert(&body, "qwen3", "").unwrap().unwrap();
        let p = internal["prompt"].as_str().unwrap();
        assert!(p.starts_with("<|im_start|>system\n# Tools") && p.contains("\"name\":\"get_weather\""), "{p}");
        assert!(p.contains("Be brief.<|im_end|>\n<|im_start|>user\nweather in Pune?<|im_end|>\n"), "{p}");
        assert!(p.contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nPune\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>"), "{p}");
        assert!(p.contains("<|im_start|>user\n<tool_response>\n31C\n</tool_response>\n<tool_response>\ndry\n</tool_response><|im_end|>"), "{p}");
        // The request's other fields ride along; the chat fields do not.
        assert_eq!((internal["temperature"].as_f64(), internal["max_tokens"].as_u64()), (Some(0.2), Some(64)));
        assert!(internal.get("messages").is_none() && internal.get("tools").is_none());
    }

    #[test]
    fn a_forced_call_starts_the_reply_inside_the_call() {
        let named = json!({"messages": [{"role": "user", "content": "x"}], "tools": weather(),
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}});
        let (internal, mut reply) = convert(&named, "qwen3", "").unwrap().unwrap();
        assert!(internal["prompt"].as_str().unwrap().ends_with("<tool_call>\n<function=get_weather>\n"));
        let generated = "<parameter=city>\nPune\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(reply.delta(generated), "");
        let (parsed, rest) = reply.finish(generated);
        assert_eq!(parsed.calls, vec![("get_weather".to_string(), json!({"city": "Pune"}))]);
        assert_eq!(rest, "");

        let required = json!({"messages": [{"role": "user", "content": "x"}], "tools": weather(), "tool_choice": "required"});
        let (internal, reply) = convert(&required, "qwen3", "").unwrap().unwrap();
        assert!(internal["prompt"].as_str().unwrap().ends_with("<tool_call>\n<function="));
        let (parsed, _) = reply.finish("get_weather>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(parsed.calls, vec![("get_weather".to_string(), json!({"days": 3}))]);
    }

    #[test]
    fn requests_that_cannot_be_served_are_refused() {
        let base = |extra: Value| {
            let mut b = json!({"messages": [{"role": "user", "content": "x"}], "tools": weather()});
            for (k, v) in extra.as_object().unwrap() { b[k] = v.clone(); }
            b
        };
        assert!(convert(&base(json!({})), "llama", "").is_err(), "another chat format");
        assert!(convert(&base(json!({"response_format": {"type": "json_object"}})), "qwen3", "").is_err());
        assert!(convert(&base(json!({"tool_choice": {"type": "function", "function": {"name": "nope"}}})), "qwen3", "").is_err());
        assert!(convert(&base(json!({"tool_choice": "sometimes"})), "qwen3", "").is_err());
    }

    #[test]
    fn text_streams_up_to_a_call_and_never_shows_part_of_its_tag() {
        let body = json!({"messages": [{"role": "user", "content": "x"}], "tools": weather()});
        let (_, mut reply) = convert(&body, "qwen3", "").unwrap().unwrap();
        let whole = "\n\nLet me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPune\n</parameter>\n</function>\n</tool_call>";
        let mut seen = String::new();
        // Token by token, in pieces that split the tag.
        let mut at = 0;
        for step in [1, 1, 4, 5, 6, 3, 2, 7, 40, 200] {
            at = (at + step).min(whole.len());
            seen.push_str(reply.delta(&whole[..at]));
            assert!(!seen.contains('<'), "{seen:?}");
        }
        assert_eq!(seen, "Let me check.");
        let (parsed, rest) = reply.finish(whole);
        assert_eq!((parsed.text.as_str(), rest.as_str()), ("Let me check.", ""));
        assert_eq!(calls_json(&parsed.calls, "ab", true)[0], json!({"id": "call_ab0", "type": "function", "index": 0,
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Pune\"}"}}));

        // A reply with no call: a "<" that turns out to be text is shown once it cannot be a tag.
        let (_, mut plain) = convert(&body, "qwen3", "").unwrap().unwrap();
        assert_eq!(plain.delta("a <"), "a");
        assert_eq!(plain.delta("a <b"), " <b");
        let (parsed, rest) = plain.finish("a <b> c ");
        assert_eq!((parsed.text.as_str(), rest.as_str()), ("a <b> c", "> c"));
    }
}
