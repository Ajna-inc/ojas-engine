//! The Anthropic Messages API (`POST /v1/messages`) over the same engine.
//!
//! A request becomes a prompt in the ChatML tool format Qwen-family templates use
//! (tools declared in the system turn, calls as `<tool_call><function=…>`, results
//! as `<tool_response>`); the generated text becomes content blocks (`thinking`,
//! `text`, `tool_use`). Streaming is buffered: a tool call can only be recognised
//! once complete, so events go out when generation ends.

use serde_json::{json, Map, Value};
use std::collections::HashMap;

/// What a reply needs to turn text back into content blocks.
pub(crate) struct Shape {
    /// Whether the model was asked to think; its output then starts inside `<think>`.
    pub(crate) thinking: bool,
    /// Declared type of each tool parameter, to type the values it is called with.
    param_types: HashMap<(String, String), String>,
}

/// A request's internal form: the prompt text and the sampling fields the engine reads.
pub(crate) struct Converted {
    pub(crate) body: Value,
    pub(crate) shape: Shape,
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks.iter().filter_map(|b| match b.get("type").and_then(Value::as_str) {
            Some("text") => b.get("text").and_then(Value::as_str).map(str::to_string),
            Some("image") => Some("[image]".to_string()),
            _ => None,
        }).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

fn arg_text(v: &Value) -> String {
    match v { Value::String(s) => s.clone(), other => other.to_string() }
}

/// Render the request in the model's chat format.
pub(crate) fn convert(body: &Value, default_system: &str) -> Result<Converted, String> {
    let msgs = body.get("messages").and_then(Value::as_array).ok_or("\"messages\" is required")?;
    let thinking = body.pointer("/thinking/type").and_then(Value::as_str) == Some("enabled");
    let tools: Vec<&Value> = if body.pointer("/tool_choice/type").and_then(Value::as_str) == Some("none") {
        Vec::new()
    } else {
        body.get("tools").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default()
    };
    let mut param_types = HashMap::new();
    let mut system = String::new();
    if thinking {
        system.push_str("Reasoning effort is set to xhigh. Please think carefully through the task, validate key \
            assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity \
            in the final answer.\n\n");
    }
    if !tools.is_empty() {
        system.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for t in &tools {
            let name = t.get("name").and_then(Value::as_str).unwrap_or_default();
            let schema = t.get("input_schema").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                for (k, p) in props {
                    if let Some(ty) = p.get("type").and_then(Value::as_str) {
                        param_types.insert((name.to_string(), k.clone()), ty.to_string());
                    }
                }
            }
            let tool = json!({"type": "function", "function": {
                "name": name, "description": t.get("description").cloned().unwrap_or(json!("")), "parameters": schema}});
            system.push('\n');
            system.push_str(&tool.to_string());
        }
        system.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO \
            suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n\
            </parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can \
            span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function \
            calls MUST follow the specified format: an inner <function=...></function> block must be nested \
            within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide \
            optional reasoning for your function call in natural language BEFORE the function call, but NOT \
            after\n- If there is no function call available, answer the question like normal with your current \
            knowledge and do not tell the user about function calls\n</IMPORTANT>");
    }
    let user_system = match body.get("system") {
        Some(s) if !s.is_null() => text_of(s),
        _ => default_system.to_string(),
    };
    let user_system = user_system.trim();
    if !user_system.is_empty() {
        if !system.is_empty() { system.push_str("\n\n"); }
        system.push_str(user_system);
    }

    let mut out = String::new();
    if !system.is_empty() { out.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n")); }
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content").unwrap_or(&Value::Null);
        let blocks: Vec<Value> = match content {
            Value::Array(a) => a.clone(),
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            _ => Vec::new(),
        };
        if role == "assistant" {
            let reasoning: String = blocks.iter().filter(|b| b["type"] == "thinking")
                .filter_map(|b| b["thinking"].as_str()).collect::<Vec<_>>().join("\n");
            let text = text_of(&Value::Array(blocks.iter().filter(|b| b["type"] == "text").cloned().collect()));
            out.push_str(&format!("<|im_start|>assistant\n<think>\n{}\n</think>\n\n{}", reasoning.trim(), text.trim()));
            let mut first = true;
            for b in blocks.iter().filter(|b| b["type"] == "tool_use") {
                out.push_str(if first && !text.trim().is_empty() { "\n\n" } else if first { "" } else { "\n" });
                first = false;
                out.push_str(&format!("<tool_call>\n<function={}>\n", b["name"].as_str().unwrap_or_default()));
                if let Some(args) = b.get("input").and_then(Value::as_object) {
                    for (k, v) in args { out.push_str(&format!("<parameter={k}>\n{}\n</parameter>\n", arg_text(v))); }
                }
                out.push_str("</function>\n</tool_call>");
            }
            out.push_str("<|im_end|>\n");
        } else {
            let mut turn = String::new();
            for b in &blocks {
                match b["type"].as_str() {
                    Some("tool_result") => turn.push_str(&format!(
                        "\n<tool_response>\n{}\n</tool_response>", text_of(b.get("content").unwrap_or(&Value::Null)).trim())),
                    Some("text") | Some("image") => {
                        if !turn.is_empty() { turn.push('\n'); }
                        turn.push_str(&text_of(&Value::Array(vec![b.clone()])));
                    }
                    _ => {}
                }
            }
            // A turn of only tool results opens with the tool's own newline, as the template does.
            out.push_str(&format!("<|im_start|>user{}{}<|im_end|>\n",
                if turn.starts_with('\n') { "" } else { "\n" }, turn));
        }
    }
    out.push_str("<|im_start|>assistant\n<think>\n");
    if !thinking { out.push_str("\n</think>\n\n"); }

    let mut internal = Map::new();
    internal.insert("prompt".into(), Value::String(out));
    for (from, to) in [("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"),
                       ("top_k", "top_k"), ("stop_sequences", "stop"), ("stream", "stream")] {
        if let Some(v) = body.get(from) { internal.insert(to.into(), v.clone()); }
    }
    internal.insert("cache_prompt".into(), Value::Bool(true));
    Ok(Converted { body: Value::Object(internal), shape: Shape { thinking, param_types } })
}

/// What a reply's text holds.
pub(crate) struct Parsed {
    pub(crate) thinking: Option<String>,
    pub(crate) text: String,
    pub(crate) calls: Vec<(String, Value)>,
}

impl Shape {
    fn typed(&self, tool: &str, key: &str, raw: &str) -> Value {
        match self.param_types.get(&(tool.to_string(), key.to_string())).map(String::as_str) {
            Some("string") | None => Value::String(raw.to_string()),
            Some(_) => serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string())),
        }
    }

    pub(crate) fn parse(&self, generated: &str) -> Parsed {
        let (thinking, rest) = match generated.find("</think>") {
            Some(i) => (Some(generated[..i].trim().to_string()), &generated[i + "</think>".len()..]),
            None if self.thinking => (Some(generated.trim().to_string()), ""),
            None => (None, generated),
        };
        let (text, mut calls) = match rest.find("<tool_call>") {
            Some(i) => (rest[..i].trim().to_string(), Vec::new()),
            None => (rest.trim().to_string(), Vec::new()),
        };
        let mut cursor = rest;
        while let Some(i) = cursor.find("<tool_call>") {
            let body = &cursor[i + "<tool_call>".len()..];
            let end = body.find("</tool_call>").unwrap_or(body.len());
            let call = &body[..end];
            if let Some(f) = call.find("<function=") {
                let after = &call[f + "<function=".len()..];
                if let Some(close) = after.find('>') {
                    let name = after[..close].trim().to_string();
                    let mut args = Map::new();
                    let mut p = &after[close + 1..];
                    while let Some(s) = p.find("<parameter=") {
                        let q = &p[s + "<parameter=".len()..];
                        let Some(c) = q.find('>') else { break };
                        let key = q[..c].trim().to_string();
                        let v = &q[c + 1..];
                        let e = v.find("</parameter>").unwrap_or(v.len());
                        let raw = &v[..e];
                        let raw = raw.strip_prefix('\n').unwrap_or(raw);
                        let raw = raw.strip_suffix('\n').unwrap_or(raw);
                        args.insert(key.clone(), self.typed(&name, &key, raw));
                        p = &v[(e + "</parameter>".len()).min(v.len())..];
                    }
                    calls.push((name, Value::Object(args)));
                }
            }
            cursor = &body[(end + "</tool_call>".len()).min(body.len())..];
        }
        Parsed { thinking: thinking.filter(|t| !t.is_empty()), text, calls }
    }
}

/// The `content` array and `stop_reason` for a parsed reply.
pub(crate) fn blocks(p: &Parsed, id: &str) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(t) = &p.thinking { out.push(json!({"type": "thinking", "thinking": t, "signature": ""})); }
    if !p.text.is_empty() || p.calls.is_empty() { out.push(json!({"type": "text", "text": p.text})); }
    for (k, (name, input)) in p.calls.iter().enumerate() {
        out.push(json!({"type": "tool_use", "id": format!("toolu_{id}{k}"), "name": name, "input": input}));
    }
    out
}

pub(crate) fn message(id: &str, model: &str, content: Vec<Value>, stop_reason: &str, input: usize, output: usize) -> Value {
    let stop_reason = if stop_reason.is_empty() { Value::Null } else { json!(stop_reason) };
    let id = id.replacen("cmpl-", "msg_", 1);
    json!({"id": id, "type": "message", "role": "assistant", "model": model, "content": content,
           "stop_reason": stop_reason, "stop_sequence": null,
           "usage": {"input_tokens": input, "output_tokens": output}})
}

pub(crate) fn event(name: &str, v: &Value) -> Vec<u8> { format!("event: {name}\ndata: {v}\n\n").into_bytes() }

/// The events of a whole reply after `message_start`.
pub(crate) fn stream_events(content: &[Value], stop_reason: &str, output: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, b) in content.iter().enumerate() {
        let (start, delta) = match b["type"].as_str() {
            Some("thinking") => (json!({"type": "thinking", "thinking": ""}),
                                 json!({"type": "thinking_delta", "thinking": b["thinking"]})),
            Some("tool_use") => (json!({"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}),
                                 json!({"type": "input_json_delta", "partial_json": b["input"].to_string()})),
            _ => (json!({"type": "text", "text": ""}), json!({"type": "text_delta", "text": b["text"]})),
        };
        out.extend(event("content_block_start", &json!({"type": "content_block_start", "index": i, "content_block": start})));
        out.extend(event("content_block_delta", &json!({"type": "content_block_delta", "index": i, "delta": delta})));
        if b["type"] == "thinking" {
            out.extend(event("content_block_delta", &json!({"type": "content_block_delta", "index": i,
                "delta": {"type": "signature_delta", "signature": ""}})));
        }
        out.extend(event("content_block_stop", &json!({"type": "content_block_stop", "index": i})));
    }
    out.extend(event("message_delta", &json!({"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null}, "usage": {"output_tokens": output}})));
    out.extend(event("message_stop", &json!({"type": "message_stop"})));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_exchange_renders_in_the_template_format() {
        let body = json!({
            "system": "Be brief.",
            "tools": [{"name": "Read", "description": "read", "input_schema": {"type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}],
            "messages": [
                {"role": "user", "content": "open it"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.rs", "limit": 5}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "fn main(){}"}]}
            ]});
        let c = convert(&body, "").unwrap();
        let p = c.body["prompt"].as_str().unwrap();
        assert!(p.starts_with("<|im_start|>system\n# Tools"));
        assert!(p.contains("\n\nBe brief.<|im_end|>\n<|im_start|>user\nopen it<|im_end|>\n"));
        assert!(p.contains("ok\n\n<tool_call>\n<function=Read>\n<parameter=path>\na.rs\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>"));
        assert!(p.contains("<|im_start|>user\n<tool_response>\nfn main(){}\n</tool_response><|im_end|>\n") ||
                p.contains("<|im_start|>user\n<tool_response>\nfn main(){}\n</tool_response><|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn calls_come_back_typed_by_their_schema() {
        let body = json!({"tools": [{"name": "Read", "input_schema": {"properties":
            {"path": {"type": "string"}, "limit": {"type": "integer"}}}}],
            "messages": [{"role": "user", "content": "x"}]});
        let shape = convert(&body, "").unwrap().shape;
        let p = shape.parse("\n\nLooking.\n\n<tool_call>\n<function=Read>\n<parameter=path>\n/a/b.rs\n</parameter>\n\
            <parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.text, "Looking.");
        assert_eq!(p.calls, vec![("Read".to_string(), json!({"path": "/a/b.rs", "limit": 20}))]);
        let b = blocks(&p, "x");
        assert_eq!(b.len(), 2);
        assert_eq!(b[1]["type"], "tool_use");
    }

    #[test]
    fn plain_text_has_one_text_block_and_thinking_is_split_off() {
        let body = json!({"thinking": {"type": "enabled"}, "messages": [{"role": "user", "content": "x"}]});
        let c = convert(&body, "").unwrap();
        assert!(c.body["prompt"].as_str().unwrap().ends_with("<|im_start|>assistant\n<think>\n"));
        let p = c.shape.parse("hmm\n</think>\n\nHello.");
        assert_eq!((p.thinking.as_deref(), p.text.as_str()), (Some("hmm"), "Hello."));
        assert!(stream_events(&blocks(&p, "x"), "end_turn", 3).starts_with(b"event: content_block_start"));
    }
}
