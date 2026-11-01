//! Capture, end to end over real TCP.
//!
//! The unit tests prove the store redacts and encrypts. These prove the gateway
//! actually *calls* it — that a real request through a real proxy lands in a real
//! file with the sensitive parts gone. The distinction matters: M6's earlier
//! sibling, shadow mirroring, was fully computed and displayed while never
//! dispatching a request. A capture path that is wired but never invoked would
//! fail in exactly the same shape.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Router as AxumRouter, http::StatusCode};
use futures_util::StreamExt;
use onpar_gateway::capture::CaptureStore;
use onpar_gateway::proxy::{AppState, app};
use onpar_gateway::router::{Backend, Phase, Route, Router};

async fn spawn(router: AxumRouter) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    addr
}

/// A plain JSON completion that echoes an email back, so redaction has work to do
/// on the response as well as the request.
async fn json_upstream() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        r#"{"choices":[{"message":{"role":"assistant","content":"Sent to ada@example.com"}}],
            "usage":{"prompt_tokens":11,"completion_tokens":4}}"#,
    )
}

/// The same content, streamed in deltas, so the transcript must be reassembled.
async fn stream_upstream() -> impl IntoResponse {
    let frames = vec![
        r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#.to_string(),
        r#"data: {"choices":[{"delta":{"content":"Sent to "}}]}"#.to_string(),
        r#"data: {"choices":[{"delta":{"content":"ada@example.com"}}]}"#.to_string(),
        r#"data: {"usage":{"prompt_tokens":11,"completion_tokens":4}}"#.to_string(),
        "data: [DONE]".to_string(),
    ];
    let s = futures_util::stream::iter(frames)
        .map(|f| Ok::<_, std::io::Error>(axum::body::Bytes::from(format!("{f}\n\n"))));
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        axum::body::Body::from_stream(s),
    )
}

fn backend(name: &str, at: SocketAddr) -> Backend {
    Backend {
        name: name.into(),
        base_url: format!("http://{at}"),
        model: None,
    }
}

/// Bring up an upstream and a gateway wired to a capture store at `log`.
async fn gateway(log: &Path, handler: axum::routing::MethodRouter) -> (SocketAddr, Arc<AppState>) {
    let up = spawn(AxumRouter::new().route("/chat/completions", handler)).await;
    let store = CaptureStore::open(log, &[3u8; 32]).unwrap();
    let state = Arc::new(
        AppState::new(
            Router::new(Route {
                phase: Phase::Off,
                incumbent: backend("incumbent", up),
                candidate: backend("candidate", up),
                failover: false,
            }),
            reqwest::Client::new(),
        )
        .with_capture(Arc::new(store)),
    );
    let addr = spawn(app(Arc::clone(&state))).await;
    (addr, state)
}

fn body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-5",
        "stream": stream,
        "messages": [{"role": "user", "content": "email ada@example.com the invoice"}],
    })
}

/// Capture is written off the request path, so a completed response does not
/// mean a completed write. Poll rather than sleep a fixed amount.
async fn wait_for(store: &CaptureStore, n: usize) -> Vec<onpar_gateway::capture::Capture> {
    for _ in 0..100 {
        let got = store.read_all().unwrap();
        if got.len() >= n {
            return got;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("capture never appeared");
}

#[tokio::test]
async fn a_non_streamed_request_is_captured_with_both_sides_redacted() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(json_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("x-request-id", "req-1")
        .json(&body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // The client still gets the real content — redaction is for the log, not the
    // response. Capturing must not alter what the caller sees.
    assert!(resp.text().await.unwrap().contains("ada@example.com"));

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let got = wait_for(&store, 1).await;
    assert_eq!(got.len(), 1);
    let c = &got[0];

    assert_eq!(c.request_id, "req-1");
    assert_eq!(c.model, "gpt-5");
    assert_eq!(c.backend, "incumbent");
    assert_eq!(c.prompt_tokens, Some(11));
    assert_eq!(c.completion_tokens, Some(4));

    assert_eq!(c.response, "Sent to <EMAIL>");
    let prompt = c.messages.to_string();
    assert!(!prompt.contains("ada@example.com"), "{prompt}");
    assert!(prompt.contains("the invoice"), "content must survive");
    assert!(!c.redacted.is_empty());
}

#[tokio::test]
async fn a_streamed_response_is_reassembled_from_its_deltas() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(stream_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&body(true))
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    while s.next().await.is_some() {}

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let got = wait_for(&store, 1).await;
    assert_eq!(
        got[0].response, "Sent to <EMAIL>",
        "deltas must be concatenated, and the role-only frame skipped"
    );
    assert_eq!(got[0].completion_tokens, Some(4));
}

#[tokio::test]
async fn the_file_on_disk_contains_no_plaintext_at_all() {
    // The strongest statement available: not just "the email is gone" but "none
    // of it is readable" — this is what NFR-3 means in practice.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(json_upstream)).await;

    reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    wait_for(&store, 1).await;

    let raw = String::from_utf8_lossy(&std::fs::read(&log).unwrap()).into_owned();
    for leak in ["ada@example.com", "the invoice", "gpt-5", "incumbent"] {
        assert!(!raw.contains(leak), "{leak:?} is readable on disk");
    }
}

#[tokio::test]
async fn a_gateway_without_a_store_writes_nothing_and_still_serves() {
    // Capture is opt-in. The default must be a proxy that records no prompts.
    let up = spawn(AxumRouter::new().route("/chat/completions", post(json_upstream))).await;
    let state = Arc::new(AppState::new(
        Router::new(Route {
            phase: Phase::Off,
            incumbent: backend("incumbent", up),
            candidate: backend("candidate", up),
            failover: false,
        }),
        reqwest::Client::new(),
    ));
    assert!(!state.capturing());

    let addr = spawn(app(Arc::clone(&state))).await;
    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(state.records().len(), 1, "metering is unaffected");
}

#[tokio::test]
async fn many_concurrent_requests_all_land_without_corrupting_each_other() {
    // Appends interleave across tasks. A partial write from one would desync the
    // frame boundaries and take the whole log with it.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(json_upstream)).await;

    let client = reqwest::Client::new();
    let mut tasks = Vec::new();
    for i in 0..25 {
        let c = client.clone();
        let url = format!("http://{addr}/v1/chat/completions");
        tasks.push(tokio::spawn(async move {
            c.post(url)
                .header("x-request-id", format!("req-{i}"))
                .json(&body(false))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let got = wait_for(&store, 25).await;
    assert_eq!(got.len(), 25);
    let mut ids: Vec<_> = got.iter().map(|c| c.request_id.clone()).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 25, "every request is present exactly once");
}

// --- the shape fields the distiller clusters on ---------------------------------

/// A response that calls a tool instead of answering — the case that was
/// recorded as an ordinary empty answer.
async fn tool_upstream() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        r#"{"choices":[{"message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c1","type":"function",
                "function":{"name":"refund","arguments":"{\"order\":\"ada@example.com\"}"}}]}}],
            "usage":{"prompt_tokens":9,"completion_tokens":3}}"#,
    )
}

/// The same, streamed: the name arrives in the opening fragment and the
/// arguments dribble in after it.
async fn tool_stream_upstream() -> impl IntoResponse {
    let frames = vec![
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"refund","arguments":""}}]}}]}"#.to_string(),
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"order\":"}}]}}]}"#.to_string(),
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ada@example.com\"}"}}]}}]}"#.to_string(),
        "data: [DONE]".to_string(),
    ];
    let s = futures_util::stream::iter(frames)
        .map(|f| Ok::<_, std::io::Error>(axum::body::Bytes::from(format!("{f}\n\n"))));
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        axum::body::Body::from_stream(s),
    )
}

fn tool_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-5",
        "stream": stream,
        "messages": [{"role": "user", "content": "refund my order"}],
        "tools": [{"type": "function", "function": {"name": "refund", "description": "issue a refund"}}],
        "response_format": {"type": "json_object"},
    })
}

#[tokio::test]
async fn a_tool_using_exchange_records_the_three_fields_it_clusters_on() {
    // Before this, `tools`, `tool_calls` and `response_format` were never
    // recorded, so `extract_shape` clustered every tool-using workload as
    // toolless — a well-formed clustering, blind along three of its six
    // dimensions, with nothing to say so.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(tool_upstream)).await;

    reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&tool_body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);

    assert_eq!(c.tool_calls.len(), 1);
    assert_eq!(c.tool_calls[0].name, "refund");
    // The argument contains the order id, which is an email address and must be redacted.
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "email must be redacted in arguments"
    );
    assert!(
        c.tool_calls[0].arguments.contains("<EMAIL>"),
        "redaction placeholder must appear"
    );
    assert_eq!(c.response_format.as_deref(), Some("json_object"));
    assert_eq!(
        c.tools.pointer("/0/function/name").and_then(|v| v.as_str()),
        Some("refund"),
        "the offered schema is what names the cluster"
    );
}

#[tokio::test]
async fn a_streamed_tool_call_is_recorded_once_from_its_opening_fragment() {
    // The name appears in one fragment and the arguments in the rest. Counting
    // fragments would report three calls to a tool that was called once.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(tool_stream_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&tool_body(true))
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    while s.next().await.is_some() {}

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);
    assert_eq!(c.tool_calls.len(), 1, "once, not per fragment");
    assert_eq!(c.tool_calls[0].name, "refund");
    // Arguments are reassembled from the 3 fragments; email is redacted.
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "email must not appear in reassembled arguments"
    );
    assert!(
        c.tool_calls[0].arguments.contains("<EMAIL>"),
        "redaction placeholder must appear in reassembled arguments"
    );
}

#[tokio::test]
async fn tool_call_arguments_are_stored_with_pii_redacted() {
    // Arguments are now stored because in a multi-turn agent loop every step's
    // `messages` already carries the previous steps' `tool_calls` with full
    // arguments — names-only cost us argument grading. The invariant is now
    // redact-before-persist, not omit: PII leaves, JSON structure survives.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(tool_upstream)).await;

    reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&tool_body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);
    assert_eq!(c.tool_calls.len(), 1);
    // PII must be gone.
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "raw email must not appear in stored arguments"
    );
    // JSON structure (the key) survives so argument grading can work.
    assert!(
        c.tool_calls[0].arguments.contains("order"),
        "JSON key must survive redaction for grading"
    );
    // The redaction placeholder is present, confirming redaction ran.
    assert!(c.tool_calls[0].arguments.contains("<EMAIL>"));
}

#[tokio::test]
async fn a_request_offering_no_tools_records_none_rather_than_guessing() {
    // The control for all three: a plain request must not acquire a tool shape.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(json_upstream)).await;

    reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);
    assert!(c.tool_calls.is_empty());
    assert!(c.response_format.is_none());
    assert!(c.tools.is_null());
}
// ── Anthropic Messages API capture ───────────────────────────────────────────

/// Non-streamed Anthropic Messages response with text and a tool_use block.
async fn messages_json_upstream() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        r#"{"id":"msg_01","type":"message","role":"assistant",
            "content":[
                {"type":"text","text":"Sent to ada@example.com"},
                {"type":"tool_use","id":"tu1","name":"notify","input":{"email":"ada@example.com"}}
            ],
            "usage":{"input_tokens":11,"output_tokens":7}}"#,
    )
}

/// Streamed Anthropic Messages response: text delta then tool_use with args.
async fn messages_stream_upstream() -> impl IntoResponse {
    let frames = vec![
        r#"data: {"type":"message_start","message":{"usage":{"input_tokens":11,"output_tokens":1}}}"#.to_string(),
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#.to_string(),
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Sent to "}}"#.to_string(),
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ada@example.com"}}"#.to_string(),
        r#"data: {"type":"content_block_stop","index":0}"#.to_string(),
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu1","name":"notify","input":{}}}"#.to_string(),
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"email\":"}}"#.to_string(),
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"ada@example.com\"}"}}"#.to_string(),
        r#"data: {"type":"content_block_stop","index":1}"#.to_string(),
        r#"data: {"type":"message_delta","delta":{},"usage":{"output_tokens":7}}"#.to_string(),
        r#"data: {"type":"message_stop"}"#.to_string(),
    ];
    let s = futures_util::stream::iter(frames)
        .map(|f| Ok::<_, std::io::Error>(axum::body::Bytes::from(format!("{f}\n\n"))));
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        axum::body::Body::from_stream(s),
    )
}

fn messages_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "claude-3-5",
        "stream": stream,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "email ada@example.com the invoice"}],
        "tools": [{"name": "notify", "description": "send a notification", "input_schema": {"type": "object"}}]
    })
}

async fn gateway_for_route(
    log: &std::path::Path,
    route: &str,
    handler: axum::routing::MethodRouter,
) -> (std::net::SocketAddr, std::sync::Arc<AppState>) {
    let up = spawn(axum::Router::new().route(route, handler)).await;
    let store = CaptureStore::open(log, &[3u8; 32]).unwrap();
    let state = std::sync::Arc::new(
        AppState::new(
            onpar_gateway::router::Router::new(onpar_gateway::router::Route {
                phase: onpar_gateway::router::Phase::Off,
                incumbent: backend("incumbent", up),
                candidate: backend("candidate", up),
                failover: false,
            }),
            reqwest::Client::new(),
        )
        .with_capture(std::sync::Arc::new(store)),
    );
    let addr = spawn(app(std::sync::Arc::clone(&state))).await;
    (addr, state)
}

#[tokio::test]
async fn messages_non_streamed_capture_normalised_to_chat_shape() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway_for_route(&log, "/messages", post(messages_json_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/messages"))
        .header("x-request-id", "msg-req-1")
        .json(&messages_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Client gets the original upstream bytes unchanged.
    let body_text = resp.text().await.unwrap();
    assert!(
        body_text.contains("ada@example.com"),
        "client must receive upstream bytes unchanged"
    );

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);

    assert_eq!(c.request_id, "msg-req-1");
    assert_eq!(c.model, "claude-3-5");
    assert_eq!(c.backend, "incumbent");
    // Tokens from Anthropic's input_tokens/output_tokens fields.
    assert_eq!(c.prompt_tokens, Some(11));
    assert_eq!(c.completion_tokens, Some(7));

    // Response text assembled from text content blocks, email redacted.
    assert!(
        !c.response.contains("ada@example.com"),
        "response email must be redacted"
    );
    assert!(
        c.response.contains("<EMAIL>"),
        "redaction placeholder must appear"
    );

    // Tool calls extracted from tool_use content blocks.
    assert_eq!(c.tool_calls.len(), 1);
    assert_eq!(c.tool_calls[0].name, "notify");
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "tool call arguments must be redacted"
    );

    // Messages normalised from Anthropic format: system + user.
    let msgs = c.messages.as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");
    assert!(
        !msgs[1]["content"]
            .as_str()
            .unwrap()
            .contains("ada@example.com"),
        "user message email must be redacted"
    );

    // Tools normalised to OpenAI function shape.
    let tools = c.tools.as_array().unwrap();
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "notify");
}

#[tokio::test]
async fn messages_streamed_capture_reassembles_text_and_tool_calls() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway_for_route(&log, "/messages", post(messages_stream_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/messages"))
        .json(&messages_body(true))
        .send()
        .await
        .unwrap();
    // Drain the stream so FinishOnDrop fires.
    let mut s = resp.bytes_stream();
    while s.next().await.is_some() {}

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);

    // Text reassembled from text_delta events, email redacted.
    assert!(
        !c.response.contains("ada@example.com"),
        "streamed response email must be redacted"
    );
    assert!(c.response.contains("<EMAIL>"));

    // Tool call reassembled from content_block_start + input_json_delta events.
    assert_eq!(c.tool_calls.len(), 1, "one tool call from streaming events");
    assert_eq!(c.tool_calls[0].name, "notify");
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "streamed tool arguments must be redacted"
    );

    // Tokens from message_start + message_delta events.
    assert_eq!(c.prompt_tokens, Some(11));
    assert_eq!(c.completion_tokens, Some(7));
}

// ── OpenAI Responses API capture ─────────────────────────────────────────────

async fn responses_json_upstream() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        r#"{"output":[
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Sent to ada@example.com"}]},
                {"type":"function_call","call_id":"fc1","name":"notify","arguments":"{\"email\":\"ada@example.com\"}"}
            ],
            "usage":{"input_tokens":10,"output_tokens":5}}"#,
    )
}

async fn responses_stream_upstream() -> impl IntoResponse {
    let frames = vec![
        r#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}"#.to_string(),
        r#"data: {"type":"response.output_text.delta","output_index":0,"delta":"Sent to "}"#.to_string(),
        r#"data: {"type":"response.output_text.delta","output_index":0,"delta":"ada@example.com"}"#.to_string(),
        r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message"}}"#.to_string(),
        r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"fc1","name":"notify","arguments":""}}"#.to_string(),
        r#"data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"email\":"}"#.to_string(),
        r#"data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"\"ada@example.com\"}"}"#.to_string(),
        r#"data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":5}}}"#.to_string(),
    ];
    let s = futures_util::stream::iter(frames)
        .map(|f| Ok::<_, std::io::Error>(axum::body::Bytes::from(format!("{f}\n\n"))));
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        axum::body::Body::from_stream(s),
    )
}

fn responses_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o",
        "stream": stream,
        "instructions": "You are helpful.",
        "input": "email ada@example.com the invoice",
        "tools": [{"type": "function", "name": "notify", "description": "send a notification", "parameters": {"type": "object"}}]
    })
}

#[tokio::test]
async fn responses_non_streamed_capture_normalised_to_chat_shape() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway_for_route(&log, "/responses", post(responses_json_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/responses"))
        .header("x-request-id", "resp-req-1")
        .json(&responses_body(false))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body_text = resp.text().await.unwrap();
    assert!(
        body_text.contains("ada@example.com"),
        "client must receive upstream bytes unchanged"
    );

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);

    assert_eq!(c.request_id, "resp-req-1");
    assert_eq!(c.model, "gpt-4o");
    assert_eq!(c.prompt_tokens, Some(10));
    assert_eq!(c.completion_tokens, Some(5));

    // Response text from output[].message.content[].output_text, email redacted.
    assert!(
        !c.response.contains("ada@example.com"),
        "response email must be redacted"
    );
    assert!(c.response.contains("<EMAIL>"));

    // Tool call from output[].function_call item.
    assert_eq!(c.tool_calls.len(), 1);
    assert_eq!(c.tool_calls[0].name, "notify");
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "tool call arguments must be redacted"
    );

    // Messages normalised: instructions → system, input string → user.
    let msgs = c.messages.as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[1]["role"], "user");

    // Tools normalised to OpenAI function shape (nested under "function").
    let tools = c.tools.as_array().unwrap();
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "notify");
}

#[tokio::test]
async fn responses_streamed_capture_reassembles_text_and_tool_calls() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) =
        gateway_for_route(&log, "/responses", post(responses_stream_upstream)).await;

    let resp = reqwest::Client::new()
        .post(format!("http://{addr}/v1/responses"))
        .json(&responses_body(true))
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    while s.next().await.is_some() {}

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let c = wait_for(&store, 1).await.remove(0);

    assert!(
        !c.response.contains("ada@example.com"),
        "streamed response email must be redacted"
    );
    assert!(c.response.contains("<EMAIL>"));

    assert_eq!(c.tool_calls.len(), 1, "one tool call from streaming events");
    assert_eq!(c.tool_calls[0].name, "notify");
    assert!(
        !c.tool_calls[0].arguments.contains("ada@example.com"),
        "streamed tool arguments must be redacted"
    );

    // Tokens from response.completed event.
    assert_eq!(c.prompt_tokens, Some(10));
    assert_eq!(c.completion_tokens, Some(5));
}

// ── Seam: both new routes go through the same capture path ───────────────────

/// A gateway that has capture enabled but does not touch it until after the
/// response — verifies the seam by checking `capturing()` mirrors what
/// `with_capture` set, then confirms the capture fired for each route.
#[tokio::test]
async fn both_new_routes_capture_via_the_same_path_as_chat_completions() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");

    // Single upstream serving all three paths.
    let up = spawn(
        axum::Router::new()
            .route("/chat/completions", post(json_upstream))
            .route("/messages", post(messages_json_upstream))
            .route("/responses", post(responses_json_upstream)),
    )
    .await;
    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let state = std::sync::Arc::new(
        AppState::new(
            onpar_gateway::router::Router::new(onpar_gateway::router::Route {
                phase: onpar_gateway::router::Phase::Off,
                incumbent: backend("incumbent", up),
                candidate: backend("candidate", up),
                failover: false,
            }),
            reqwest::Client::new(),
        )
        .with_capture(std::sync::Arc::new(store)),
    );
    assert!(state.capturing(), "capture must be on");

    let gw_addr = spawn(app(std::sync::Arc::clone(&state))).await;
    let client = reqwest::Client::new();

    client
        .post(format!("http://{gw_addr}/v1/chat/completions"))
        .json(&body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    client
        .post(format!("http://{gw_addr}/v1/messages"))
        .json(&messages_body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    client
        .post(format!("http://{gw_addr}/v1/responses"))
        .json(&responses_body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let got = wait_for(&store, 3).await;
    assert_eq!(got.len(), 3, "all three routes must produce a capture");
    // All three Captures must have been stored (non-empty model).
    for c in &got {
        assert!(!c.model.is_empty(), "every capture must have a model");
        assert!(!c.backend.is_empty(), "every capture must have a backend");
    }
}

#[test]
fn ready_fails_on_a_log_path_that_cannot_be_written() {
    // `open` builds a cipher and touches no filesystem; `append` opens lazily,
    // from a spawned task whose errors are logged and dropped. So a caller that
    // checked only `open` got a gateway which started, served traffic, and
    // recorded nothing — indistinguishable from one that was working.
    let dir = tempfile::tempdir().expect("tempdir");
    let occupied = dir.path().join("log-is-a-directory");
    std::fs::create_dir(&occupied).expect("mkdir");

    let store = CaptureStore::open(&occupied, &[3u8; 32]).expect("open must still succeed");
    assert!(
        store.ready().is_err(),
        "a directory is not an appendable log"
    );
}

#[test]
fn ready_succeeds_and_does_not_write_a_record() {
    // The control: a readiness probe that appended something would corrupt the
    // corpus it is checking, and one that always failed would be useless.
    let dir = tempfile::tempdir().expect("tempdir");
    let log = dir.path().join("captures.log");

    let store = CaptureStore::open(&log, &[4u8; 32]).expect("open");
    store.ready().expect("a fresh path must be writable");

    assert!(log.exists(), "ready opens the log");
    assert_eq!(
        store.read_all().expect("read").len(),
        0,
        "ready must not record anything"
    );
}

#[tokio::test]
async fn captures_without_a_request_id_header_get_distinct_non_empty_ids() {
    // Before this fix `request_id` was `unwrap_or_default()`, so every headerless
    // capture landed with `request_id: ""` and the whole log aliased to one id.
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("captures");
    let (addr, _state) = gateway(&log, post(json_upstream)).await;

    let client = reqwest::Client::new();
    // Two requests with no x-request-id header.
    for _ in 0..2 {
        client
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&body(false))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
    }
    // One request with an explicit header.
    client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("x-request-id", "explicit-id")
        .json(&body(false))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let store = CaptureStore::open(&log, &[3u8; 32]).unwrap();
    let got = wait_for(&store, 3).await;
    assert_eq!(got.len(), 3);

    // Records arrive in write order (off-path tasks), not send order. Search by content.
    let explicit = got
        .iter()
        .find(|c| c.request_id == "explicit-id")
        .expect("record with supplied header must be present");
    assert_eq!(
        explicit.request_id, "explicit-id",
        "supplied header must be kept verbatim"
    );

    let generated: Vec<_> = got
        .iter()
        .filter(|c| c.request_id != "explicit-id")
        .collect();
    assert_eq!(generated.len(), 2);
    assert!(
        !generated[0].request_id.is_empty(),
        "generated id must not be empty"
    );
    assert!(
        !generated[1].request_id.is_empty(),
        "generated id must not be empty"
    );
    assert_ne!(
        generated[0].request_id, generated[1].request_id,
        "two headerless captures must have distinct ids"
    );
}
