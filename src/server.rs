/// OpenAI-compatible HTTP server that proxies to the active model backend.
///
/// Routes:
///   GET  /health                   — liveness check
///   GET  /v1/models                — list registered models (OpenAI format)
///   POST /v1/chat/completions      — chat inference (streaming + non-streaming)
///   POST /v1/completions           — legacy completion (maps to chat)
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
};
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

use crate::config::Config;
use crate::registry::{ModelEntry, ModelSource, Registry};

// ── Server state file ─────────────────────────────────────────────────────────
// Written to disk so the TUI and CLI can detect that the server is running.

#[derive(Debug, Serialize, Deserialize)]
pub struct ServeStateFile {
    pub port: u16,
    pub active_model: String,
    pub pid: u32,
}

pub fn read_serve_state() -> Option<ServeStateFile> {
    let path = crate::config::server_state_path().ok()?;
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn write_serve_state(port: u16, active_model: &str) {
    if let Ok(path) = crate::config::server_state_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let state = ServeStateFile {
            port,
            active_model: active_model.to_string(),
            pid: std::process::id(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&state) {
            let _ = std::fs::write(&path, json);
        }
    }
}

fn delete_serve_state() {
    if let Ok(path) = crate::config::server_state_path() {
        let _ = std::fs::remove_file(path);
    }
}

// ── Shared state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ServerState {
    pub config: Config,
    pub registry: Registry,
}

// ── OpenAI wire types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    #[allow(dead_code)]
    pub temperature: f32,
    pub max_tokens: Option<u32>,
}

fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct StreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelObject>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn gen_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn router(state: ServerState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health",               get(health))
        .route("/v1/models",            get(list_models))
        .route("/v1/chat/completions",  post(chat_completions))
        .route("/v1/completions",       post(legacy_completions))
        .with_state(Arc::new(state))
        .layer(cors)
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health(State(s): State<Arc<ServerState>>) -> impl IntoResponse {
    let active = s.registry.models.iter().find(|m| m.active).map(|m| m.name.as_str()).unwrap_or("none");
    Json(serde_json::json!({ "status": "ok", "active_model": active }))
}

async fn list_models(State(s): State<Arc<ServerState>>) -> impl IntoResponse {
    let data: Vec<ModelObject> = s.registry.models.iter().map(|m| ModelObject {
        id: m.name.clone(),
        object: "model",
        created: 0,
        owned_by: m.backend.clone(),
    }).collect();
    Json(ModelList { object: "list", data })
}

async fn legacy_completions(
    State(s): State<Arc<ServerState>>,
    headers: HeaderMap,
    Json(mut req): Json<serde_json::Value>,
) -> Response {
    // Convert legacy /v1/completions to chat format
    let prompt = req["prompt"].as_str().unwrap_or("").to_string();
    req["messages"] = serde_json::json!([{"role": "user", "content": prompt}]);
    let chat_req: ChatRequest = match serde_json::from_value(req) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    chat_completions(State(s), headers, Json(chat_req)).await
}

async fn chat_completions(
    State(s): State<Arc<ServerState>>,
    _headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    // Resolve model: from request, or registry active model
    // Resolve model name while still holding a borrow on `s`
    let model_name_resolved = match resolve_model(&s, req.model.as_deref()) {
        Ok(m) => m.name.clone(),
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    // Drop the borrow before moving `s` into the response helpers
    let model = match s.registry.get(&model_name_resolved) {
        Some(m) => m.clone(),
        None => return (StatusCode::INTERNAL_SERVER_ERROR, "model vanished".to_string()).into_response(),
    };

    if req.stream {
        stream_response_owned(s, req, model).await
    } else {
        blocking_response_owned(s, req, model).await
    }
}

// ── Model resolution ──────────────────────────────────────────────────────────

fn resolve_model<'a>(s: &'a ServerState, name: Option<&str>) -> Result<&'a ModelEntry> {
    match name {
        Some(n) => s.registry.get(n)
            .ok_or_else(|| anyhow::anyhow!("model '{n}' not found in registry")),
        None => s.registry.models.iter().find(|m| m.active)
            .ok_or_else(|| anyhow::anyhow!(
                "no active model — set one with: deepsage switch <model>"
            )),
    }
}

// ── Non-streaming response ────────────────────────────────────────────────────

async fn blocking_response_owned(
    s: Arc<ServerState>,
    req: ChatRequest,
    model: ModelEntry,
) -> Response {
    match do_infer(&s.config, &model, &req.messages, false, req.max_tokens).await {
        Ok(content) => {
            let resp = ChatResponse {
                id: gen_id(),
                object: "chat.completion",
                created: now(),
                model: model.name.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage { role: "assistant".into(), content },
                    finish_reason: "stop",
                }],
                usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            };
            Json(resp).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}})),
        ).into_response(),
    }
}

// ── Streaming response (SSE) ──────────────────────────────────────────────────

async fn stream_response_owned(
    s: Arc<ServerState>,
    req: ChatRequest,
    model: ModelEntry,
) -> Response {
    let id = gen_id();
    let model_name = model.name.clone();
    let cfg = s.config.clone();

    let stream = stream_infer(cfg, model, req.messages, id, model_name);
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

fn stream_infer(
    cfg: Config,
    model: ModelEntry,
    messages: Vec<ChatMessage>,
    id: String,
    model_name: String,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    async_stream::stream! {
        // Opening delta with role
        let open = StreamChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created: now(),
            model: model_name.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&open).unwrap()));

        match model.backend.as_str() {
            "ollama" => {
                let tag = match &model.source {
                    ModelSource::Ollama { tag } => tag.clone(),
                    _ => model.name.clone(),
                };
                let url = format!("{}/api/chat", cfg.ollama.url.trim_end_matches('/'));
                let body = serde_json::json!({
                    "model": tag,
                    "messages": messages,
                    "stream": true,
                });
                let client = reqwest::Client::new();
                match client.post(&url).json(&body).send().await {
                    Err(e) => {
                        let err = StreamChunk {
                            id: id.clone(), object: "chat.completion.chunk",
                            created: now(), model: model_name.clone(),
                            choices: vec![StreamChoice {
                                index: 0,
                                delta: StreamDelta { role: None, content: Some(format!("[error: {e}]")) },
                                finish_reason: Some("stop"),
                            }],
                        };
                        yield Ok(Event::default().data(serde_json::to_string(&err).unwrap()));
                    }
                    Ok(resp) => {
                        let mut byte_stream = resp.bytes_stream();
                        while let Some(chunk) = byte_stream.next().await {
                            let chunk = match chunk { Ok(c) => c, Err(_) => break };
                            for line in chunk.split(|&b| b == b'\n') {
                                if line.is_empty() { continue; }
                                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                                    let text = v["message"]["content"].as_str().unwrap_or("").to_string();
                                    let done = v["done"].as_bool().unwrap_or(false);
                                    if !text.is_empty() || done {
                                        let chunk = StreamChunk {
                                            id: id.clone(), object: "chat.completion.chunk",
                                            created: now(), model: model_name.clone(),
                                            choices: vec![StreamChoice {
                                                index: 0,
                                                delta: StreamDelta { role: None, content: Some(text) },
                                                finish_reason: if done { Some("stop") } else { None },
                                            }],
                                        };
                                        yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
                                        if done { break; }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            "llamacpp" => {
                // llama-server already speaks OpenAI streaming — proxy directly
                let url = format!("http://{}:{}/v1/chat/completions",
                    cfg.llamacpp.host, cfg.llamacpp.port);
                let body = serde_json::json!({
                    "model": model.name,
                    "messages": messages,
                    "stream": true,
                });
                let client = reqwest::Client::new();
                if let Ok(resp) = client.post(&url).json(&body).send().await {
                    let mut byte_stream = resp.bytes_stream();
                    while let Some(chunk) = byte_stream.next().await {
                        let chunk = match chunk { Ok(c) => c, Err(_) => break };
                        for line in String::from_utf8_lossy(&chunk).lines() {
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" { break; }
                                yield Ok(Event::default().data(data));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // SSE terminator
        yield Ok(Event::default().data("[DONE]"));
    }
}

// ── Shared non-streaming inference helper ─────────────────────────────────────

async fn do_infer(
    cfg: &Config,
    model: &ModelEntry,
    messages: &[ChatMessage],
    _stream: bool,
    _max_tokens: Option<u32>,
) -> Result<String> {
    let client = reqwest::Client::new();
    match model.backend.as_str() {
        "ollama" => {
            let tag = match &model.source {
                ModelSource::Ollama { tag } => tag.clone(),
                _ => model.name.clone(),
            };
            let url = format!("{}/api/chat", cfg.ollama.url.trim_end_matches('/'));
            let body = serde_json::json!({
                "model": tag,
                "messages": messages,
                "stream": false,
            });
            let resp: serde_json::Value = client.post(&url).json(&body).send().await?.json().await?;
            if let Some(err) = resp["error"].as_str() {
                anyhow::bail!("Ollama error: {}", err);
            }
            let content = resp["message"]["content"].as_str().unwrap_or("").to_string();
            if content.is_empty() {
                anyhow::bail!("Ollama returned empty content. Raw response: {}", resp);
            }
            Ok(content)
        }
        "llamacpp" => {
            let url = format!("http://{}:{}/v1/chat/completions",
                cfg.llamacpp.host, cfg.llamacpp.port);
            let body = serde_json::json!({
                "model": model.name,
                "messages": messages,
                "stream": false,
            });
            let resp: serde_json::Value = client.post(&url).json(&body).send().await?.json().await?;
            Ok(resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
        }
        other => anyhow::bail!("unknown backend '{other}'"),
    }
}

// ── Kill-on-drop wrapper for child processes ──────────────────────────────────

struct KillOnDrop(std::process::Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

pub async fn serve(host: &str, port: u16, cfg: Config) -> Result<()> {
    let registry = crate::registry::load().unwrap_or_default();
    let active = registry.models.iter().find(|m| m.active)
        .map(|m| m.name.clone())
        .unwrap_or_else(|| "none".into());

    // Auto-spawn llama-server when the active model uses the llamacpp backend
    let _llama_guard: Option<KillOnDrop> =
        if let Some(model) = registry.models.iter().find(|m| m.active) {
            if model.backend == "llamacpp" {
                let local_path = model.local_path.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "model '{}' has no local GGUF file.\n\
                         Run `deepsage pick` to download and register a model.",
                        model.name
                    )
                })?;

                let binary =
                    crate::backends::llamacpp::resolve_binary(&cfg.llamacpp.server_binary)
                        .ok_or_else(|| anyhow::anyhow!(
                            "llama-server not found.\n\
                             Install with:  brew install llama.cpp"
                        ))?;

                let n_gpu_layers: u32 = if model.vram_alloc_gb > 0.0 { 999 } else { 0 };
                let internal_port = cfg.llamacpp.port;

                println!("\x1b[2m  Spawning llama-server for '{}' on internal port {}…\x1b[0m",
                    model.name, internal_port);

                let child = crate::backends::llamacpp::spawn_server(
                    &binary,
                    std::path::Path::new(local_path),
                    "127.0.0.1",
                    internal_port,
                    n_gpu_layers,
                    4096,
                )?;
                let guard = KillOnDrop(child);

                print!("\x1b[2m  Waiting for llama-server to be ready");
                use std::io::Write;
                std::io::stdout().flush()?;

                if !crate::backends::llamacpp::wait_for_ready(
                    "127.0.0.1", internal_port, 120,
                ).await {
                    anyhow::bail!(
                        "llama-server did not become ready within 120 seconds.\n\
                         Check that the model file is valid and llama-server is working."
                    );
                }
                println!(" ✓\x1b[0m");

                Some(guard)
            } else {
                None
            }
        } else {
            None
        };

    let state = ServerState { config: cfg, registry };
    let app = router(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Write state file so TUI / CLI can detect the running server
    write_serve_state(port, &active);

    println!("\x1b[1m\x1b[36m DeepSage Inference Server\x1b[0m");
    println!("{}", "─".repeat(60));
    println!("  Listening on  \x1b[1mhttp://{addr}\x1b[0m");
    println!("  Active model  \x1b[1m{active}\x1b[0m");
    println!();
    println!("  \x1b[2mOpenAI-compatible endpoints:\x1b[0m");
    println!("    GET  http://{addr}/v1/models");
    println!("    POST http://{addr}/v1/chat/completions");
    println!("    GET  http://{addr}/health");
    println!();
    println!("  \x1b[2mPython (openai SDK):\x1b[0m");
    println!("    from openai import OpenAI");
    println!("    client = OpenAI(base_url=\"http://{addr}/v1\", api_key=\"none\")");
    println!("    resp = client.chat.completions.create(");
    println!("        model=\"{active}\",");
    println!("        messages=[{{\"role\": \"user\", \"content\": \"Hello!\"}}]");
    println!("    )");
    println!("    print(resp.choices[0].message.content)");
    println!();
    println!("  \x1b[2mPython (requests):\x1b[0m");
    println!("    import requests");
    println!("    r = requests.post(\"http://{addr}/v1/chat/completions\", json={{");
    println!("        \"model\": \"{active}\",");
    println!("        \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]");
    println!("    }})");
    println!("    print(r.json()[\"choices\"][0][\"message\"][\"content\"])");
    println!();
    println!("  \x1b[2mPress Ctrl-C to stop.\x1b[0m");
    println!("{}", "─".repeat(60));

    axum::serve(listener, app)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); })
        .await?;

    // Clean up: delete state file and kill llama-server
    delete_serve_state();
    drop(_llama_guard);
    Ok(())
}
