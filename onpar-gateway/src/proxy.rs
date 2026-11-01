//! The HTTP datapath.
//!
//! Three routes share the same backend selection, metering, and capture path:
//!
//! - `POST /v1/chat/completions` — OpenAI chat
//! - `POST /v1/messages`         — Anthropic Messages API
//! - `POST /v1/responses`        — OpenAI Responses API
//!
//! Each route is a thin wrapper around `handle_route`, which carries a
//! `ShapeOps` struct describing how to extract capture fields from that API's
//! wire format. Bytes are forwarded to the upstream unchanged; normalisation
//! happens only in the capture layer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures_util::StreamExt;
use parking_lot::Mutex;

use crate::capture::normalize;
use crate::capture::store::{self, Capture, CaptureStore, ToolCall};
use crate::meter::{Meter, Metered};
use crate::router::{Decision, Router};
use crate::sse::{Decoder, Event};

/// Per-process random prefix for generated request IDs.
static REQUEST_ID_PREFIX: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
    use chacha20poly1305::aead::rand_core::RngCore;
    chacha20poly1305::aead::OsRng.next_u32()
});

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_request_id() -> String {
    let n = REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("gw-{:08x}-{n:06}", *REQUEST_ID_PREFIX)
}

/// `fn(&[u8]) -> (Option<u64>, Option<u64>)` — token extractor for a body.
type BodyTokensFn = fn(&[u8]) -> (Option<u64>, Option<u64>);
/// `fn(&Event) -> Option<(index, name, args_fragment)>` — tool-call fragment extractor.
type DeltaToolCallFn = fn(&Event) -> Option<(usize, Option<String>, String)>;
/// `fn(&Event) -> (Option<u64>, Option<u64>)` — token extractor for a stream event.
type StreamTokensFn = fn(&Event) -> (Option<u64>, Option<u64>);

/// Extraction functions for one API shape.
///
/// All fields are `Copy` (fn pointers + `&'static str`), so `ShapeOps` can be
/// captured cheaply by the streaming closure.
#[derive(Clone, Copy)]
struct ShapeOps {
    /// Path appended to the backend base URL when forwarding, e.g. `"chat/completions"`.
    upstream_path: &'static str,
    /// Normalise the request body to the capture fields the distiller reads.
    extract_request: fn(&serde_json::Value) -> normalize::RequestFields,
    /// Extract response text from a complete (non-streamed) response body.
    body_text: fn(&[u8]) -> String,
    /// Extract tool calls from a complete (non-streamed) response body.
    body_tool_calls: fn(&[u8]) -> Vec<ToolCall>,
    /// Extract token counts (prompt, completion) from a complete response body.
    body_tokens: BodyTokensFn,
    /// Extract text from a single streamed event.
    delta_text: fn(&Event) -> Option<String>,
    /// Extract a tool-call fragment from a single streamed event.
    delta_tool_call: DeltaToolCallFn,
    /// Extract token counts from a single streamed event.
    ///
    /// Returns `(None, None)` for API shapes where the `Meter` handles tokens
    /// via the standard OpenAI `usage` key (chat completions).
    stream_tokens: StreamTokensFn,
}

impl ShapeOps {
    fn chat() -> Self {
        Self {
            upstream_path: "chat/completions",
            extract_request: normalize::chat_request_fields,
            body_text: store::body_text,
            body_tool_calls: store::body_tool_calls,
            body_tokens: normalize::chat_body_tokens,
            delta_text: store::delta_text,
            delta_tool_call: store::delta_tool_call,
            stream_tokens: normalize::chat_stream_tokens,
        }
    }

    fn messages() -> Self {
        Self {
            upstream_path: "messages",
            extract_request: normalize::messages_request_fields,
            body_text: normalize::messages_body_text,
            body_tool_calls: normalize::messages_body_tool_calls,
            body_tokens: normalize::messages_body_tokens,
            delta_text: normalize::messages_delta_text,
            delta_tool_call: normalize::messages_delta_tool_call,
            stream_tokens: normalize::messages_stream_tokens,
        }
    }

    fn responses() -> Self {
        Self {
            upstream_path: "responses",
            extract_request: normalize::responses_request_fields,
            body_text: normalize::responses_body_text,
            body_tool_calls: normalize::responses_body_tool_calls,
            body_tokens: normalize::responses_body_tokens,
            delta_text: normalize::responses_delta_text,
            delta_tool_call: normalize::responses_delta_tool_call,
            stream_tokens: normalize::responses_stream_tokens,
        }
    }
}

/// Shared server state.
pub struct AppState {
    /// Current routing table, swapped atomically on reconfiguration.
    pub router: parking_lot::RwLock<Router>,
    /// Shared HTTP client used for all upstream requests.
    pub client: reqwest::Client,
    records: Mutex<Vec<Record>>,
    mirrors: Mutex<Vec<MirrorRecord>>,
    /// Ring-buffer capacity for the in-memory record log.
    pub max_records: usize,
    transitions: Mutex<Vec<crate::control::Transition>>,
    /// Bearer token required for admin endpoints; `None` disables auth.
    pub admin_token: Option<String>,
    capture: Option<Arc<CaptureStore>>,
}

/// What one request cost and where it went.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Record {
    /// Backend that served the response.
    pub backend: String,
    /// True when the primary was unreachable and the fallback answered.
    pub failed_over: bool,
    /// Backend the request was mirrored to, if any.
    pub mirrored_to: Option<String>,
    /// Why the router chose this path.
    pub reason: &'static str,
    /// Model the client asked for.
    pub model: String,
    /// Whether the client asked for a stream.
    pub streaming: bool,
    /// Upstream status code.
    pub status: u16,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Token accounting, or an explicit statement that upstream reported none.
    pub metered: Metered,
}

/// What a mirrored request did. Written independently of the client's response,
/// because the mirror must never delay or influence it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MirrorRecord {
    /// Candidate backend the request was mirrored to.
    pub backend: String,
    /// Model the client asked for.
    pub model: String,
    /// Status the candidate returned, or `None` if it could not be reached.
    pub status: Option<u16>,
    /// How long the candidate took.
    pub duration_ms: u64,
    /// Token accounting for the candidate's response.
    pub metered: Metered,
    /// Transport failure, if any.
    pub error: Option<String>,
}

impl AppState {
    /// New state with a default record cap.
    pub fn new(router: Router, client: reqwest::Client) -> Self {
        Self {
            router: parking_lot::RwLock::new(router),
            client,
            records: Mutex::new(Vec::new()),
            mirrors: Mutex::new(Vec::new()),
            max_records: 10_000,
            transitions: Mutex::new(Vec::new()),
            admin_token: crate::control::token_from_env(),
            capture: None,
        }
    }

    /// Persist redacted traffic to `store`.
    #[must_use]
    pub fn with_capture(mut self, store: Arc<CaptureStore>) -> Self {
        self.capture = Some(store);
        self
    }

    /// Whether traffic is being persisted.
    pub fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Write one exchange off the request path.
    fn capture(&self, c: Capture) {
        let Some(store) = self.capture.clone() else {
            return;
        };
        tokio::spawn(async move {
            match tokio::task::spawn_blocking(move || store.append(c)).await {
                Ok(Ok(report)) => tracing::debug!(redacted = report.total(), "capture stored"),
                Ok(Err(e)) => tracing::warn!(error = %e, "capture not stored"),
                Err(e) => tracing::warn!(error = %e, "capture task failed"),
            }
        });
    }

    /// Append a request record to the in-memory ring buffer, evicting the oldest if full.
    pub fn record(&self, r: Record) {
        let mut g = self.records.lock();
        if g.len() >= self.max_records {
            g.remove(0);
        }
        tracing::info!(
            backend = %r.backend,
            model = %r.model,
            status = r.status,
            duration_ms = r.duration_ms,
            reported = r.metered.is_reported(),
            tokens = r.metered.usage().map_or(0, |u| u.total()),
            "request complete"
        );
        g.push(r);
    }

    /// Snapshot of retained records.
    pub fn records(&self) -> Vec<Record> {
        self.records.lock().clone()
    }

    /// Record a completed mirror, dropping the oldest once the cap is reached.
    pub fn record_mirror(&self, m: MirrorRecord) {
        let mut g = self.mirrors.lock();
        if g.len() >= self.max_records {
            g.remove(0);
        }
        tracing::info!(
            backend = %m.backend,
            status = m.status.unwrap_or(0),
            duration_ms = m.duration_ms,
            error = m.error.as_deref().unwrap_or(""),
            "mirror complete"
        );
        g.push(m);
    }

    /// Snapshot of retained mirror outcomes.
    pub fn mirrors(&self) -> Vec<MirrorRecord> {
        self.mirrors.lock().clone()
    }

    /// Record an applied phase change.
    pub fn record_transition(&self, t: crate::control::Transition) {
        self.transitions.lock().push(t);
    }

    /// Every phase change applied to this process, oldest first.
    pub fn transitions(&self) -> Vec<crate::control::Transition> {
        self.transitions.lock().clone()
    }
}

fn dispatch_mirror(
    st: Arc<AppState>,
    backend: crate::router::Backend,
    parsed: serde_json::Value,
    headers: HeaderMap,
    body: Bytes,
    model: String,
    upstream_path: &'static str,
) {
    tokio::spawn(async move {
        let started = Instant::now();
        let decision = Decision {
            serve: backend.clone(),
            mirror: None,
            fallback: None,
            reason: "mirror",
        };
        let mut rec = MirrorRecord {
            backend: backend.name.clone(),
            model,
            status: None,
            duration_ms: 0,
            metered: Metered::Unreported { deltas: 0 },
            error: None,
        };

        match send(
            &st,
            &decision.serve,
            upstream_path,
            &parsed,
            &headers,
            &body,
        )
        .await
        {
            Ok(resp) => {
                rec.status = Some(resp.status().as_u16());
                let mut meter = Meter::new();
                let mut decoder = Decoder::new();
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(b) => {
                            for ev in decoder.push(&b) {
                                meter.observe(&ev);
                            }
                            meter.observe_body(&b);
                        }
                        Err(e) => {
                            rec.error = Some(e.to_string());
                            break;
                        }
                    }
                }
                rec.metered = meter.finish();
            }
            Err(e) => rec.error = Some(e.to_string()),
        }
        rec.duration_ms = elapsed_ms(started);
        st.record_mirror(rec);
    });
}

const CONSOLE_HTML: &str = include_str!("console.html");

/// Build the router. Kept separate from `serve` so tests can drive it directly.
pub fn app(state: Arc<AppState>) -> AxumRouter {
    AxumRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/responses", post(openai_responses))
        .route("/healthz", get(|| async { "ok" }))
        .route("/metrics/requests", get(records))
        .route("/metrics/mirrors", get(mirrors))
        .route(
            "/control/phase",
            get(crate::control::get_phase).post(crate::control::set_phase),
        )
        .route("/control/history", get(crate::control::history))
        .route("/metrics/engine", get(engine_telemetry))
        .route("/", get(console))
        .with_state(state)
}

async fn console() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        CONSOLE_HTML,
    )
}

async fn records(State(st): State<Arc<AppState>>) -> Json<Vec<Record>> {
    Json(st.records())
}

async fn mirrors(State(st): State<Arc<AppState>>) -> Json<Vec<MirrorRecord>> {
    Json(st.mirrors())
}

async fn engine_telemetry(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let backends: Vec<(String, String)> = {
        let r = st.router.read();
        let route = r.route_for(None);
        let mut seen = std::collections::BTreeMap::new();
        for b in [&route.incumbent, &route.candidate] {
            seen.insert(b.name.clone(), b.base_url.clone());
        }
        seen.into_iter().collect()
    };

    let client = &st.client;
    let readings = futures_util::future::join_all(backends.iter().map(|(name, url)| async move {
        (name.clone(), crate::telemetry::scrape(client, url).await)
    }))
    .await;

    let host = tokio::task::spawn_blocking(crate::host::probe)
        .await
        .unwrap_or(crate::host::Support::Unavailable {
            reason: "host probe task failed".into(),
        });
    let host_peak = host.peak_memory_used();
    let sys = crate::host::system();
    let starved = host.starved_by_host(&sys);

    Json(serde_json::json!({
        "source": crate::telemetry::SOURCE,
        "host": host,
        "system": sys,
        "host_starvation": starved,
        "backends": readings
            .into_iter()
            .map(|(name, snap)| {
                let mut against_plan = snap.contradictions(
                    snap.prefix_hit_rate.is_some(),
                    snap.draft_acceptance.is_some(),
                );
                if let Some(foreign) = host.foreign_memory(snap.kv_cache_used) {
                    against_plan.push(foreign);
                }
                serde_json::json!({
                    "backend": name,
                    "telemetry": snap,
                    "derived": {
                        "kv_pressure": snap.kv_pressure(),
                        "truncation_rate": snap.truncation_rate(),
                        "abandon_rate": snap.abandon_rate(),
                        "prefill_share": snap.prefill_share(),
                        "host_memory_used": host_peak,
                    },
                    "contradicts_plan": against_plan,
                })
            })
            .collect::<Vec<_>>(),
    }))
}

/// Errors that the proxy layer can return to the caller.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// The request body could not be parsed or was structurally invalid.
    #[error("invalid request body: {0}")]
    BadRequest(String),
    /// The upstream backend returned an error or was unreachable.
    #[error("upstream {backend} unreachable: {source}")]
    Upstream {
        /// The base URL of the backend that failed.
        backend: String,
        /// The underlying reqwest transport or HTTP error.
        #[source]
        source: reqwest::Error,
    },
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Upstream { .. } => (StatusCode::BAD_GATEWAY, "upstream_error"),
        };
        let body = serde_json::json!({
            "error": { "message": self.to_string(), "type": kind }
        });
        (status, Json(body)).into_response()
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    handle_route(st, headers, body, ShapeOps::chat()).await
}

async fn anthropic_messages(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    handle_route(st, headers, body, ShapeOps::messages()).await
}

async fn openai_responses(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ProxyError> {
    handle_route(st, headers, body, ShapeOps::responses()).await
}

// ── Shared handler ────────────────────────────────────────────────────────────

async fn handle_route(
    st: Arc<AppState>,
    headers: HeaderMap,
    body: Bytes,
    ops: ShapeOps,
) -> Result<Response, ProxyError> {
    let started = Instant::now();

    let parsed: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| ProxyError::BadRequest(format!("expected JSON: {e}")))?;
    let model = parsed
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_owned();
    let streaming = parsed
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let cluster = headers.get("x-onpar-cluster").and_then(|v| v.to_str().ok());
    let key = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&model);

    let decision = st.router.read().decide(cluster, key);
    let span = tracing::info_span!(
        "proxy",
        route = ops.upstream_path,
        backend = %decision.serve.name,
        model = %model,
        streaming
    );
    let _e = span.enter();

    if let Some(candidate) = decision.mirror.clone() {
        dispatch_mirror(
            Arc::clone(&st),
            candidate,
            parsed.clone(),
            headers.clone(),
            body.clone(),
            model.clone(),
            ops.upstream_path,
        );
    }

    let mut served_by = decision.serve.name.clone();
    let mut failed_over = false;
    let upstream = match send(
        &st,
        &decision.serve,
        ops.upstream_path,
        &parsed,
        &headers,
        &body,
    )
    .await
    {
        Ok(r) => r,
        Err(primary_err) => match &decision.fallback {
            Some(alt) => {
                tracing::warn!(
                    primary = %decision.serve.name,
                    fallback = %alt.name,
                    error = %primary_err,
                    "primary unreachable, falling over"
                );
                failed_over = true;
                served_by = alt.name.clone();
                send(&st, alt, ops.upstream_path, &parsed, &headers, &body).await?
            }
            None => return Err(primary_err),
        },
    };
    let status = upstream.status();

    let mut out_headers = HeaderMap::new();
    if let Some(ct) = upstream.headers().get(header::CONTENT_TYPE) {
        out_headers.insert(header::CONTENT_TYPE, ct.clone());
    }
    if streaming {
        out_headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        out_headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    }
    out_headers.insert(
        "x-onpar-backend",
        HeaderValue::from_str(&served_by).unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );

    let pending_capture = st.capturing().then(|| {
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(next_request_id);
        let rf = (ops.extract_request)(&parsed);
        Capture {
            request_id,
            model: model.clone(),
            backend: served_by.clone(),
            messages: rf.messages,
            response: String::new(),
            prompt_tokens: None,
            completion_tokens: None,
            duration_ms: 0,
            tools: rf.tools,
            tool_calls: Vec::new(),
            response_format: rf.response_format,
            redacted: crate::capture::Report::default(),
        }
    });

    let base = Record {
        backend: served_by.clone(),
        failed_over,
        mirrored_to: decision.mirror.as_ref().map(|m| m.name.clone()),
        reason: decision.reason,
        model,
        streaming,
        status: status.as_u16(),
        duration_ms: 0,
        metered: Metered::Unreported { deltas: 0 },
    };

    if !streaming {
        let bytes = upstream.bytes().await.map_err(|e| ProxyError::Upstream {
            backend: decision.serve.name.clone(),
            source: e,
        })?;
        let mut meter = Meter::new();
        meter.observe_body(&bytes);
        let metered = meter.finish();
        if let Some(mut c) = pending_capture {
            c.response = (ops.body_text)(&bytes);
            c.tool_calls = (ops.body_tool_calls)(&bytes);
            c.duration_ms = elapsed_ms(started);
            (c.prompt_tokens, c.completion_tokens) = (ops.body_tokens)(&bytes);
            st.capture(c);
        }
        st.record(Record {
            duration_ms: elapsed_ms(started),
            metered,
            ..base
        });
        return Ok((status, out_headers, bytes).into_response());
    }

    let st2 = Arc::clone(&st);
    let meter = Arc::new(Mutex::new(Meter::new()));
    let meter_for_stream = Arc::clone(&meter);
    let transcript = pending_capture
        .is_some()
        .then(|| Arc::new(Mutex::new(Transcript::default())));
    let transcript_for_stream = transcript.clone();
    let mut decoder = Decoder::new();
    let stream = upstream.bytes_stream().map(move |chunk| match chunk {
        Ok(bytes) => {
            let events = decoder.push(&bytes);
            if !events.is_empty() {
                let mut m = meter_for_stream.lock();
                for ev in &events {
                    m.observe(ev);
                }
                if let Some(t) = &transcript_for_stream {
                    let mut t = t.lock();
                    for ev in &events {
                        if let Some(s) = (ops.delta_text)(ev) {
                            t.text.push_str(&s);
                        }
                        if let Some((index, name, args)) = (ops.delta_tool_call)(ev) {
                            let entry = t.calls.entry(index).or_insert((None, String::new()));
                            if entry.0.is_none() {
                                entry.0 = name;
                            }
                            entry.1.push_str(&args);
                        }
                        // Accumulate tokens from APIs that embed them in stream events
                        // (Messages: message_start + message_delta; Responses: response.completed).
                        // For chat completions, stream_tokens always returns (None, None) and
                        // the Meter handles token counting via the standard `usage` key.
                        let (pt, ct) = (ops.stream_tokens)(ev);
                        if let Some(p) = pt {
                            t.prompt_tokens = Some(p);
                        }
                        if let Some(c) = ct {
                            t.completion_tokens = Some(c);
                        }
                    }
                }
            }
            Ok(bytes)
        }
        Err(e) => {
            tracing::warn!(error = %e, "upstream stream error");
            Err(std::io::Error::other(e))
        }
    });

    let counted = FinishOnDrop {
        state: st2,
        record: Mutex::new(Some(base)),
        meter,
        started,
        capture: Mutex::new(pending_capture),
        transcript,
    };
    let body = Body::from_stream(stream.chain(futures_util::stream::once(async move {
        drop(counted);
        Ok(Bytes::new())
    })));

    Ok((status, out_headers, body).into_response())
}

struct FinishOnDrop {
    state: Arc<AppState>,
    record: Mutex<Option<Record>>,
    meter: Arc<Mutex<Meter>>,
    started: Instant,
    capture: Mutex<Option<Capture>>,
    transcript: Option<Arc<Mutex<Transcript>>>,
}

/// What the stream closure accumulates for a capture.
#[derive(Default)]
struct Transcript {
    /// Assistant text, reassembled from content deltas.
    text: String,
    /// Partial tool calls keyed by index (name, accumulated_arguments_json).
    calls: std::collections::BTreeMap<usize, (Option<String>, String)>,
    /// Prompt-token count extracted directly from the stream, when the API
    /// embeds it there (Anthropic Messages, OpenAI Responses). `None` for
    /// chat completions — the `Meter` handles that via the standard `usage` key.
    prompt_tokens: Option<u64>,
    /// Completion-token count extracted directly from the stream.
    completion_tokens: Option<u64>,
}

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        let metered = self.meter.lock().snapshot();
        if let Some(mut r) = self.record.lock().take() {
            r.duration_ms = elapsed_ms(self.started);
            r.metered = metered;
            self.state.record(r);
        }
        if let Some(mut c) = self.capture.lock().take() {
            if let Some(t) = &self.transcript {
                let t = t.lock();
                c.response = t.text.clone();
                c.tool_calls = t
                    .calls
                    .values()
                    .map(|(name, args)| ToolCall {
                        name: name.clone().unwrap_or_default(),
                        arguments: args.clone(),
                    })
                    .collect();
                // Prefer tokens extracted directly from stream events (Messages /
                // Responses APIs); fall back to the Meter for chat completions.
                let meter_tokens = tokens(&metered);
                c.prompt_tokens = t.prompt_tokens.or(meter_tokens.0);
                c.completion_tokens = t.completion_tokens.or(meter_tokens.1);
            }
            c.duration_ms = elapsed_ms(self.started);
            self.state.capture(c);
        }
    }
}

fn tokens(m: &Metered) -> (Option<u64>, Option<u64>) {
    m.usage().map_or((None, None), |u| {
        (Some(u.prompt_tokens), Some(u.completion_tokens))
    })
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

async fn send(
    st: &AppState,
    backend: &crate::router::Backend,
    upstream_path: &str,
    parsed: &serde_json::Value,
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<reqwest::Response, ProxyError> {
    let payload = match &backend.model {
        Some(m) => {
            let mut v = parsed.clone();
            if let Some(obj) = v.as_object_mut() {
                obj.insert("model".into(), serde_json::Value::String(m.clone()));
            }
            Bytes::from(serde_json::to_vec(&v).unwrap_or_else(|_| body.to_vec()))
        }
        None => body.clone(),
    };

    let mut req = st
        .client
        .post(format!(
            "{}/{upstream_path}",
            backend.base_url.trim_end_matches('/')
        ))
        .header(header::CONTENT_TYPE, "application/json")
        .body(payload);

    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        req = req.header(header::AUTHORIZATION, auth.clone());
    }

    req.send().await.map_err(|e| ProxyError::Upstream {
        backend: backend.name.clone(),
        source: e,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::capture::store::{ToolCall, delta_tool_call};
    use crate::sse::Event;

    #[test]
    fn streaming_fragments_reassemble_two_calls_in_index_order() {
        let payloads = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"search","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"store","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hello\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"v\":1}"}}]}}]}"#,
        ];
        let mut t = Transcript::default();
        for payload in &payloads {
            let ev = Event::Data(payload.to_string());
            if let Some((index, name, args)) = delta_tool_call(&ev) {
                let entry = t.calls.entry(index).or_insert((None, String::new()));
                if entry.0.is_none() {
                    entry.0 = name;
                }
                entry.1.push_str(&args);
            }
        }
        let calls: Vec<ToolCall> = t
            .calls
            .values()
            .map(|(name, args)| ToolCall {
                name: name.clone().unwrap_or_default(),
                arguments: args.clone(),
            })
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"q":"hello"}"#);
        assert_eq!(calls[1].name, "store");
        assert_eq!(calls[1].arguments, r#"{"v":1}"#);
    }
}
