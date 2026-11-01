//! Normalise Anthropic Messages and OpenAI Responses API payloads into the
//! OpenAI chat-completions shape the distiller and eval pipeline read.
//!
//! All functions are pure: they read JSON values or raw bytes, return owned
//! data, and never touch the forwarded wire bytes.
//!
//! Unknown or malformed content block types are skipped with a
//! `tracing::warn!` rather than a panic or a dropped connection — the proxy
//! must always complete the request even when capture degrades.

use serde_json::{Map, Value};

use crate::capture::store::ToolCall;
use crate::sse::Event;

/// Capture-relevant fields extracted from a request body, normalised to the
/// OpenAI chat-completions shape that the distiller and eval pipeline consume.
pub struct RequestFields {
    /// OpenAI-shaped `messages` array.
    pub messages: Value,
    /// OpenAI-shaped `tools` array, or `Value::Null` when the request offered none.
    pub tools: Value,
    /// Output format the request demanded, when present.
    pub response_format: Option<String>,
}

// ── Chat completions (pass-through) ──────────────────────────────────────────

/// Extract capture fields from an OpenAI chat-completions request body.
///
/// The chat path already speaks the target shape; this is a structural pass-through
/// rather than a translation.
pub fn chat_request_fields(body: &Value) -> RequestFields {
    RequestFields {
        messages: body.get("messages").cloned().unwrap_or(Value::Null),
        tools: body.get("tools").cloned().unwrap_or(Value::Null),
        response_format: body
            .get("response_format")
            .and_then(|f| f.get("type").or(Some(f)))
            .and_then(|t| t.as_str())
            .map(ToOwned::to_owned),
    }
}

/// Extract token counts from an OpenAI chat-completions non-streamed response body.
pub fn chat_body_tokens(body: &[u8]) -> (Option<u64>, Option<u64>) {
    fn extract(body: &[u8]) -> Option<(u64, u64)> {
        let v: Value = serde_json::from_slice(body).ok()?;
        let u = v.get("usage")?;
        let p = u.get("prompt_tokens")?.as_u64()?;
        let c = u.get("completion_tokens")?.as_u64()?;
        Some((p, c))
    }
    extract(body).map_or((None, None), |(p, c)| (Some(p), Some(c)))
}

/// Chat-completions streaming events carry the standard `usage` key, so the
/// `Meter` handles them. This returns `(None, None)` always; `FinishOnDrop`
/// falls back to `tokens(&metered)` for chat.
pub fn chat_stream_tokens(_ev: &Event) -> (Option<u64>, Option<u64>) {
    (None, None)
}

// ── Anthropic Messages API ────────────────────────────────────────────────────

/// Normalise an Anthropic Messages API request body to the OpenAI chat shape.
pub fn messages_request_fields(body: &Value) -> RequestFields {
    let mut msgs: Vec<Value> = Vec::new();

    // `system` string or list-of-text-blocks → leading system message.
    if let Some(sys) = body.get("system") {
        let text = anthropic_system_text(sys);
        if !text.is_empty() {
            msgs.push(serde_json::json!({"role": "system", "content": text}));
        }
    }

    if let Some(Value::Array(messages)) = body.get("messages") {
        for msg in messages {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            push_anthropic_message(&mut msgs, role, msg.get("content"));
        }
    }

    RequestFields {
        messages: Value::Array(msgs),
        tools: anthropic_tools(body),
        response_format: None, // Anthropic Messages has no response_format field
    }
}

fn anthropic_system_text(sys: &Value) -> String {
    match sys {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|b| {
                (b.get("type").and_then(Value::as_str) == Some("text"))
                    .then(|| b.get("text").and_then(Value::as_str))
                    .flatten()
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => {
            tracing::warn!("anthropic messages: unrecognised system field shape; skipping");
            String::new()
        }
    }
}

fn push_anthropic_message(out: &mut Vec<Value>, role: &str, content: Option<&Value>) {
    match content {
        None | Some(Value::Null) => {
            out.push(serde_json::json!({"role": role, "content": ""}));
        }
        Some(Value::String(s)) => {
            out.push(serde_json::json!({"role": role, "content": s}));
        }
        Some(Value::Array(blocks)) => {
            let has_tool_use = blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"));
            let has_tool_result = blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"));

            if has_tool_result {
                // Each tool_result block → a separate role:tool message.
                for block in blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                {
                    let id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned();
                    let text = tool_result_content_text(block);
                    out.push(
                        serde_json::json!({"role": "tool", "tool_call_id": id, "content": text}),
                    );
                }
            } else if has_tool_use {
                // assistant message with tool_calls + optional leading text content.
                let mut text_parts: Vec<&str> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(t) = block.get("text").and_then(Value::as_str) {
                                text_parts.push(t);
                            }
                        }
                        Some("tool_use") => {
                            tool_calls.push(anthropic_tool_use_to_call(block));
                        }
                        Some(other) => {
                            tracing::warn!(
                                kind = other,
                                "anthropic messages: unknown block type in tool_use message; skipping"
                            );
                        }
                        None => {}
                    }
                }
                let joined = text_parts.join("\n");
                let mut m = serde_json::json!({"role": "assistant", "tool_calls": tool_calls});
                if !joined.is_empty()
                    && let Some(obj) = m.as_object_mut()
                {
                    obj.insert("content".into(), Value::String(joined));
                }
                out.push(m);
            } else {
                // Plain content: join text blocks, skip other block types.
                let text = blocks
                    .iter()
                    .filter_map(|b| {
                        (b.get("type").and_then(Value::as_str) == Some("text"))
                            .then(|| b.get("text").and_then(Value::as_str))
                            .flatten()
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                out.push(serde_json::json!({"role": role, "content": text}));
            }
        }
        Some(_) => {
            tracing::warn!("anthropic messages: unrecognised content shape; skipping message");
        }
    }
}

fn tool_result_content_text(block: &Value) -> String {
    match block.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                (p.get("type").and_then(Value::as_str) == Some("text"))
                    .then(|| p.get("text").and_then(Value::as_str))
                    .flatten()
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(_) => {
            tracing::warn!("anthropic messages: unrecognised tool_result content shape");
            String::new()
        }
    }
}

fn anthropic_tool_use_to_call(block: &Value) -> Value {
    let id = block
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    // `input` is always a JSON object per the Anthropic spec; `.to_string()` on
    // a `serde_json::Value` is infallible (no I/O, writes to a String buffer).
    let arguments = block
        .get("input")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()))
        .to_string();
    serde_json::json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments}
    })
}

fn anthropic_tools(body: &Value) -> Value {
    let Some(Value::Array(tools)) = body.get("tools") else {
        return Value::Null;
    };
    Value::Array(
        tools
            .iter()
            .map(|t| {
                let mut f = Map::new();
                if let Some(v) = t.get("name") {
                    f.insert("name".into(), v.clone());
                }
                if let Some(v) = t.get("description") {
                    f.insert("description".into(), v.clone());
                }
                f.insert(
                    "parameters".into(),
                    t.get("input_schema").cloned().unwrap_or(Value::Null),
                );
                serde_json::json!({"type": "function", "function": Value::Object(f)})
            })
            .collect(),
    )
}

/// Assistant text from an Anthropic Messages non-streamed response body.
pub fn messages_body_text(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return String::new();
    };
    let Some(Value::Array(content)) = v.get("content") else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|b| {
            (b.get("type").and_then(Value::as_str) == Some("text"))
                .then(|| b.get("text").and_then(Value::as_str))
                .flatten()
                .map(ToOwned::to_owned)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Tool calls from an Anthropic Messages non-streamed response body.
pub fn messages_body_tool_calls(body: &[u8]) -> Vec<ToolCall> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(Value::Array(content)) = v.get("content") else {
        return Vec::new();
    };
    content
        .iter()
        .filter_map(|b| {
            if b.get("type").and_then(Value::as_str) == Some("tool_use") {
                let name = b.get("name").and_then(Value::as_str)?.to_owned();
                let arguments = b
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()))
                    .to_string();
                Some(ToolCall { name, arguments })
            } else {
                None
            }
        })
        .collect()
}

/// Token counts from an Anthropic Messages non-streamed response body.
///
/// Anthropic uses `input_tokens`/`output_tokens`; these map to the
/// `prompt_tokens`/`completion_tokens` fields in `Capture`.
pub fn messages_body_tokens(body: &[u8]) -> (Option<u64>, Option<u64>) {
    fn extract(body: &[u8]) -> Option<(u64, u64)> {
        let v: Value = serde_json::from_slice(body).ok()?;
        let u = v.get("usage")?;
        Some((
            u.get("input_tokens")?.as_u64()?,
            u.get("output_tokens")?.as_u64()?,
        ))
    }
    extract(body).map_or((None, None), |(p, c)| (Some(p), Some(c)))
}

/// Text delta from an Anthropic Messages streaming event.
///
/// Only `content_block_delta` events with `delta.type: text_delta` carry text.
pub fn messages_delta_text(ev: &Event) -> Option<String> {
    let Event::Data(payload) = ev else {
        return None;
    };
    let v: Value = serde_json::from_str(payload).ok()?;
    if v.get("type").and_then(Value::as_str) != Some("content_block_delta") {
        return None;
    }
    let delta = v.get("delta")?;
    if delta.get("type").and_then(Value::as_str) != Some("text_delta") {
        return None;
    }
    let text = delta.get("text").and_then(Value::as_str)?;
    (!text.is_empty()).then(|| text.to_owned())
}

/// Tool-call fragment from an Anthropic Messages streaming event.
///
/// `content_block_start` with `type: tool_use` carries the name (index → name,
/// empty args). Subsequent `content_block_delta` with `type: input_json_delta`
/// carry argument slices (index → None, partial_json). The caller accumulates
/// fragments by index, as for the chat-completions path.
pub fn messages_delta_tool_call(ev: &Event) -> Option<(usize, Option<String>, String)> {
    let Event::Data(payload) = ev else {
        return None;
    };
    let v: Value = serde_json::from_str(payload).ok()?;
    let ev_type = v.get("type").and_then(Value::as_str)?;
    let index = v
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
        .unwrap_or(0);

    match ev_type {
        "content_block_start" => {
            let block = v.get("content_block")?;
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                return None;
            }
            let name = block.get("name").and_then(Value::as_str)?.to_owned();
            Some((index, Some(name), String::new()))
        }
        "content_block_delta" => {
            let delta = v.get("delta")?;
            if delta.get("type").and_then(Value::as_str) != Some("input_json_delta") {
                return None;
            }
            let partial = delta
                .get("partial_json")
                .and_then(Value::as_str)
                .unwrap_or("");
            Some((index, None, partial.to_owned()))
        }
        _ => None,
    }
}

/// Token counts from Anthropic Messages streaming events.
///
/// `message_start` carries prompt tokens; `message_delta` carries completion
/// tokens. These arrive in separate events so both are needed.
pub fn messages_stream_tokens(ev: &Event) -> (Option<u64>, Option<u64>) {
    let Event::Data(payload) = ev else {
        return (None, None);
    };
    let Ok(v) = serde_json::from_str::<Value>(payload) else {
        return (None, None);
    };
    match v.get("type").and_then(Value::as_str) {
        Some("message_start") => {
            let input = v
                .get("message")
                .and_then(|m| m.get("usage"))
                .and_then(|u| u.get("input_tokens"))
                .and_then(Value::as_u64);
            (input, None)
        }
        Some("message_delta") => {
            let output = v
                .get("usage")
                .and_then(|u| u.get("output_tokens"))
                .and_then(Value::as_u64);
            (None, output)
        }
        _ => (None, None),
    }
}

// ── OpenAI Responses API ──────────────────────────────────────────────────────

/// Normalise an OpenAI Responses API request body to the OpenAI chat shape.
pub fn responses_request_fields(body: &Value) -> RequestFields {
    let mut msgs: Vec<Value> = Vec::new();

    if let Some(Value::String(s)) = body.get("instructions")
        && !s.is_empty()
    {
        msgs.push(serde_json::json!({"role": "system", "content": s}));
    }

    match body.get("input") {
        Some(Value::String(s)) => {
            msgs.push(serde_json::json!({"role": "user", "content": s}));
        }
        Some(Value::Array(items)) => {
            for item in items {
                push_responses_item(&mut msgs, item);
            }
        }
        Some(_) => {
            tracing::warn!("openai responses: unrecognised input field shape");
        }
        None => {}
    }

    // `text.format.type` is where Responses puts the output format.
    let response_format = body
        .get("text")
        .and_then(|t| t.get("format"))
        .and_then(|f| f.get("type"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    RequestFields {
        messages: Value::Array(msgs),
        tools: responses_tools(body),
        response_format,
    }
}

fn push_responses_item(out: &mut Vec<Value>, item: &Value) {
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
    match item_type {
        "message" => {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = responses_message_content(item.get("content"));
            out.push(serde_json::json!({"role": role, "content": content}));
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let arguments = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            out.push(serde_json::json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }]
            }));
        }
        "function_call_output" => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let output = item
                .get("output")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            out.push(
                serde_json::json!({"role": "tool", "tool_call_id": call_id, "content": output}),
            );
        }
        other => {
            tracing::warn!(
                kind = other,
                "openai responses: skipping unknown input item type"
            );
        }
    }
}

fn responses_message_content(content: Option<&Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("input_text") | Some("output_text") => {
                    p.get("text").and_then(Value::as_str).map(ToOwned::to_owned)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(_) => {
            tracing::warn!("openai responses: unrecognised message content shape");
            String::new()
        }
    }
}

fn responses_tools(body: &Value) -> Value {
    let Some(Value::Array(tools)) = body.get("tools") else {
        return Value::Null;
    };
    // Responses tools are flat: `{"type":"function","name":"...","parameters":{...}}`
    // → nest name/description/parameters under a `function` key to match chat shape.
    Value::Array(
        tools
            .iter()
            .map(|t| {
                let mut f = Map::new();
                if let Some(v) = t.get("name") {
                    f.insert("name".into(), v.clone());
                }
                if let Some(v) = t.get("description") {
                    f.insert("description".into(), v.clone());
                }
                if let Some(v) = t.get("parameters") {
                    f.insert("parameters".into(), v.clone());
                }
                serde_json::json!({"type": "function", "function": Value::Object(f)})
            })
            .collect(),
    )
}

/// Assistant text from an OpenAI Responses non-streamed response body.
pub fn responses_body_text(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return String::new();
    };
    let Some(Value::Array(output)) = v.get("output") else {
        return String::new();
    };
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .map(|item| responses_message_content(item.get("content")))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Tool calls from an OpenAI Responses non-streamed response body.
pub fn responses_body_tool_calls(body: &[u8]) -> Vec<ToolCall> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(Value::Array(output)) = v.get("output") else {
        return Vec::new();
    };
    output
        .iter()
        .filter_map(|item| {
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                let name = item.get("name").and_then(Value::as_str)?.to_owned();
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                Some(ToolCall { name, arguments })
            } else {
                None
            }
        })
        .collect()
}

/// Token counts from an OpenAI Responses non-streamed response body.
pub fn responses_body_tokens(body: &[u8]) -> (Option<u64>, Option<u64>) {
    fn extract(body: &[u8]) -> Option<(u64, u64)> {
        let v: Value = serde_json::from_slice(body).ok()?;
        let u = v.get("usage")?;
        Some((
            u.get("input_tokens")?.as_u64()?,
            u.get("output_tokens")?.as_u64()?,
        ))
    }
    extract(body).map_or((None, None), |(p, c)| (Some(p), Some(c)))
}

/// Text delta from an OpenAI Responses streaming event.
pub fn responses_delta_text(ev: &Event) -> Option<String> {
    let Event::Data(payload) = ev else {
        return None;
    };
    let v: Value = serde_json::from_str(payload).ok()?;
    if v.get("type").and_then(Value::as_str) != Some("response.output_text.delta") {
        return None;
    }
    let text = v.get("delta").and_then(Value::as_str)?;
    (!text.is_empty()).then(|| text.to_owned())
}

/// Tool-call fragment from an OpenAI Responses streaming event.
///
/// `response.output_item.added` with `item.type: function_call` carries the name.
/// `response.function_call_arguments.delta` carries argument slices.
pub fn responses_delta_tool_call(ev: &Event) -> Option<(usize, Option<String>, String)> {
    let Event::Data(payload) = ev else {
        return None;
    };
    let v: Value = serde_json::from_str(payload).ok()?;
    let ev_type = v.get("type").and_then(Value::as_str)?;

    match ev_type {
        "response.output_item.added" => {
            let item = v.get("item")?;
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                return None;
            }
            let index = v
                .get("output_index")
                .and_then(Value::as_u64)
                .and_then(|i| usize::try_from(i).ok())
                .unwrap_or(0);
            let name = item.get("name").and_then(Value::as_str)?.to_owned();
            Some((index, Some(name), String::new()))
        }
        "response.function_call_arguments.delta" => {
            let index = v
                .get("output_index")
                .and_then(Value::as_u64)
                .and_then(|i| usize::try_from(i).ok())
                .unwrap_or(0);
            let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
            Some((index, None, delta.to_owned()))
        }
        _ => None,
    }
}

/// Token counts from an OpenAI Responses streaming event.
///
/// Both counts arrive together in `response.completed`.
pub fn responses_stream_tokens(ev: &Event) -> (Option<u64>, Option<u64>) {
    let Event::Data(payload) = ev else {
        return (None, None);
    };
    let Ok(v) = serde_json::from_str::<Value>(payload) else {
        return (None, None);
    };
    if v.get("type").and_then(Value::as_str) != Some("response.completed") {
        return (None, None);
    }
    let usage = match v.get("response").and_then(|r| r.get("usage")) {
        Some(u) => u,
        None => return (None, None),
    };
    (
        usage.get("input_tokens").and_then(Value::as_u64),
        usage.get("output_tokens").and_then(Value::as_u64),
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(s: &str) -> Event {
        Event::Data(s.to_owned())
    }

    // ── Chat pass-through ─────────────────────────────────────────────────────

    #[test]
    fn chat_request_fields_passes_messages_tools_and_format() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "response_format": {"type": "json_object"}
        });
        let f = chat_request_fields(&body);
        assert_eq!(f.messages[0]["role"], "user");
        assert_eq!(f.tools[0]["type"], "function");
        assert_eq!(f.response_format.as_deref(), Some("json_object"));
    }

    #[test]
    fn chat_body_tokens_reads_prompt_and_completion() {
        let body = serde_json::to_vec(&json!({
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }))
        .unwrap();
        assert_eq!(chat_body_tokens(&body), (Some(10), Some(20)));
    }

    #[test]
    fn chat_body_tokens_absent_usage_returns_none_pair() {
        assert_eq!(chat_body_tokens(b"{}"), (None, None));
    }

    // ── Anthropic Messages request ────────────────────────────────────────────

    #[test]
    fn anthropic_system_string_becomes_leading_system_message() {
        let body = json!({
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn anthropic_system_block_list_joins_text_blocks() {
        let body = json!({
            "system": [
                {"type": "text", "text": "Line one."},
                {"type": "image", "source": {}},
                {"type": "text", "text": "Line two."}
            ],
            "messages": []
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Line one.\nLine two.");
    }

    #[test]
    fn anthropic_string_content_passes_through() {
        let body = json!({
            "messages": [{"role": "user", "content": "hello world"}]
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["content"], "hello world");
    }

    #[test]
    fn anthropic_text_block_list_joins_text_skipping_other_types() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "part1"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "part2"}
            ]}]
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["content"], "part1\npart2");
    }

    #[test]
    fn anthropic_tool_use_blocks_become_assistant_tool_calls() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "hello"}}
            ]}]
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["role"], "assistant");
        let tc = &msgs[0]["tool_calls"][0];
        assert_eq!(tc["id"], "tu1");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "search");
        let args: Value =
            serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["q"], "hello");
    }

    #[test]
    fn anthropic_tool_result_blocks_each_become_a_tool_message() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "result A"},
                {"type": "tool_result", "tool_use_id": "tu2", "content": "result B"}
            ]}]
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "tu1");
        assert_eq!(msgs[0]["content"], "result A");
        assert_eq!(msgs[1]["tool_call_id"], "tu2");
    }

    #[test]
    fn anthropic_tool_result_with_block_list_content_joins_text() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu2", "content": [
                    {"type": "text", "text": "part A"},
                    {"type": "text", "text": "part B"}
                ]}
            ]}]
        });
        let f = messages_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["content"], "part A\npart B");
    }

    #[test]
    fn anthropic_tools_normalised_to_openai_function_shape() {
        let body = json!({
            "tools": [{
                "name": "search",
                "description": "Search the web",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
            }],
            "messages": []
        });
        let f = messages_request_fields(&body);
        let tools = f.tools.as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "search");
        assert_eq!(tools[0]["function"]["description"], "Search the web");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn anthropic_response_text_blocks_joined_tool_use_skipped() {
        let body = serde_json::to_vec(&json!({
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {}},
                {"type": "text", "text": "world"}
            ]
        }))
        .unwrap();
        assert_eq!(messages_body_text(&body), "Hello\nworld");
    }

    #[test]
    fn anthropic_response_tool_calls_extracted_with_arguments_as_json_string() {
        let body = serde_json::to_vec(&json!({
            "content": [
                {"type": "text", "text": "I'll search."},
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "Rust"}}
            ]
        }))
        .unwrap();
        let calls = messages_body_tool_calls(&body);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["q"], "Rust");
    }

    #[test]
    fn anthropic_body_tokens_read_from_input_output_keys() {
        let body = serde_json::to_vec(&json!({"usage": {"input_tokens": 25, "output_tokens": 50}}))
            .unwrap();
        assert_eq!(messages_body_tokens(&body), (Some(25), Some(50)));
    }

    #[test]
    fn anthropic_streaming_text_delta_extracted() {
        let e = ev(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
        );
        assert_eq!(messages_delta_text(&e), Some("hello".into()));
    }

    #[test]
    fn anthropic_streaming_non_text_delta_ignored() {
        let e = ev(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{"}}"#,
        );
        assert!(messages_delta_text(&e).is_none());
    }

    #[test]
    fn anthropic_streaming_tool_call_name_from_content_block_start() {
        let e = ev(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu1","name":"search","input":{}}}"#,
        );
        let (idx, name, args) = messages_delta_tool_call(&e).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(name.as_deref(), Some("search"));
        assert_eq!(args, "");
    }

    #[test]
    fn anthropic_streaming_tool_call_args_from_input_json_delta() {
        let e = ev(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"hi\"}"}}"#,
        );
        let (idx, name, args) = messages_delta_tool_call(&e).unwrap();
        assert_eq!(idx, 1);
        assert!(name.is_none());
        assert_eq!(args, r#"{"q":"hi"}"#);
    }

    #[test]
    fn anthropic_streaming_message_start_carries_prompt_tokens() {
        let e = ev(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}"#,
        );
        assert_eq!(messages_stream_tokens(&e), (Some(25), None));
    }

    #[test]
    fn anthropic_streaming_message_delta_carries_completion_tokens() {
        let e = ev(r#"{"type":"message_delta","delta":{},"usage":{"output_tokens":50}}"#);
        assert_eq!(messages_stream_tokens(&e), (None, Some(50)));
    }

    // ── OpenAI Responses request ──────────────────────────────────────────────

    #[test]
    fn responses_instructions_becomes_system_message() {
        let body = json!({"instructions": "Be helpful.", "input": "Hi"});
        let f = responses_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Be helpful.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hi");
    }

    #[test]
    fn responses_input_string_becomes_user_message() {
        let body = json!({"input": "Hello"});
        let f = responses_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Hello");
    }

    #[test]
    fn responses_input_array_normalised_to_chat_messages() {
        let body = json!({
            "input": [
                {"type": "message", "role": "user", "content": "What is 2+2?"},
                {"type": "function_call", "call_id": "fc1", "name": "calc", "arguments": "{\"expr\":\"2+2\"}"},
                {"type": "function_call_output", "call_id": "fc1", "output": "4"}
            ]
        });
        let f = responses_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "calc");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["content"], "4");
    }

    #[test]
    fn responses_message_with_part_list_joins_text_parts() {
        let body = json!({
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Hello"},
                {"type": "input_text", "text": "World"}
            ]}]
        });
        let f = responses_request_fields(&body);
        let msgs = f.messages.as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Hello\nWorld");
    }

    #[test]
    fn responses_tools_nested_under_function_key() {
        let body = json!({
            "tools": [{
                "type": "function",
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"}
            }],
            "input": "hi"
        });
        let f = responses_request_fields(&body);
        let tools = f.tools.as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "search");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn responses_text_format_type_extracted() {
        let body = json!({"text": {"format": {"type": "json_schema"}}, "input": "hi"});
        let f = responses_request_fields(&body);
        assert_eq!(f.response_format.as_deref(), Some("json_schema"));
    }

    #[test]
    fn responses_body_text_from_output_message_items() {
        let body = serde_json::to_vec(&json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]},
                {"type": "function_call", "name": "search", "arguments": "{}"}
            ]
        }))
        .unwrap();
        assert_eq!(responses_body_text(&body), "Hello!");
    }

    #[test]
    fn responses_body_tool_calls_from_function_call_output_items() {
        let body = serde_json::to_vec(&json!({
            "output": [
                {"type": "function_call", "name": "search", "arguments": "{\"q\":\"rust\"}"},
                {"type": "message", "content": [{"type": "output_text", "text": "done"}]}
            ]
        }))
        .unwrap();
        let calls = responses_body_tool_calls(&body);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn responses_body_tokens_read_from_input_output_keys() {
        let body = serde_json::to_vec(&json!({"usage": {"input_tokens": 30, "output_tokens": 60}}))
            .unwrap();
        assert_eq!(responses_body_tokens(&body), (Some(30), Some(60)));
    }

    #[test]
    fn responses_streaming_text_delta_extracted() {
        let e = ev(r#"{"type":"response.output_text.delta","output_index":0,"delta":"hello"}"#);
        assert_eq!(responses_delta_text(&e), Some("hello".into()));
    }

    #[test]
    fn responses_streaming_tool_call_name_from_output_item_added() {
        let e = ev(
            r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","name":"search","arguments":""}}"#,
        );
        let (idx, name, args) = responses_delta_tool_call(&e).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(name.as_deref(), Some("search"));
        assert_eq!(args, "");
    }

    #[test]
    fn responses_streaming_tool_call_args_from_function_call_arguments_delta() {
        let e = ev(
            r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"q\":\"hi\"}"}"#,
        );
        let (idx, name, args) = responses_delta_tool_call(&e).unwrap();
        assert_eq!(idx, 1);
        assert!(name.is_none());
        assert_eq!(args, r#"{"q":"hi"}"#);
    }

    #[test]
    fn responses_streaming_completed_carries_both_token_counts() {
        let e = ev(
            r#"{"type":"response.completed","response":{"usage":{"input_tokens":25,"output_tokens":50}}}"#,
        );
        assert_eq!(responses_stream_tokens(&e), (Some(25), Some(50)));
    }

    #[test]
    fn unknown_events_return_none_without_panicking() {
        let e = ev(r#"{"type":"totally.unknown","data":"whatever"}"#);
        assert!(messages_delta_text(&e).is_none());
        assert!(messages_delta_tool_call(&e).is_none());
        assert_eq!(messages_stream_tokens(&e), (None, None));
        assert!(responses_delta_text(&e).is_none());
        assert!(responses_delta_tool_call(&e).is_none());
        assert_eq!(responses_stream_tokens(&e), (None, None));
    }

    #[test]
    fn malformed_json_in_events_returns_none_without_panicking() {
        let e = ev("not json at all {{{{");
        assert!(messages_delta_text(&e).is_none());
        assert!(messages_delta_tool_call(&e).is_none());
        assert_eq!(messages_stream_tokens(&e), (None, None));
        assert!(responses_delta_text(&e).is_none());
        assert!(responses_delta_tool_call(&e).is_none());
        assert_eq!(responses_stream_tokens(&e), (None, None));
    }
}
