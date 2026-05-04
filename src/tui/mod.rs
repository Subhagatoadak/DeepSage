mod app;
mod ui;

pub use app::App;
use app::{ChatMessage, ChatToken};

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, time::Duration};

use crate::{config::Config, monitor, registry};

pub async fn run(config: Config) -> Result<()> {
    let reg = registry::load().unwrap_or_default();
    let mut app = App::new(config, reg);

    // Initial data load before entering TUI
    refresh_data(&mut app).await;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = event_loop(&mut terminal, &mut app).await;

    // Always restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        // Drain any streamed chat tokens before drawing (non-blocking)
        drain_chat_tokens(app);

        terminal.draw(|f| ui::draw(f, app))?;

        // Poll for input with 50 ms timeout; shorter = more responsive streaming
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                app.handle_key(key);
            }
        }

        if app.should_quit {
            break;
        }

        // Spawn chat inference task when user submitted a message.
        // The task sends AssistantStart before its first token, so we do NOT
        // pre-create an assistant message here.
        if let Some(messages) = app.pending_chat_send.take() {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ChatToken>();
            app.chat_rx = Some(rx);
            app.chat_scroll = u16::MAX / 2;
            let cfg = app.config.clone();
            let reg = app.registry.clone();
            tokio::spawn(do_chat_stream(messages, tx, cfg, reg));
        }

        // Periodic data refresh
        if app.should_refresh() {
            refresh_data(app).await;
        }

        // Clear stale status messages after 4 seconds
        if let Some((_, ts)) = &app.status_msg {
            if ts.elapsed() > Duration::from_secs(4) {
                app.status_msg = None;
            }
        }
    }
    Ok(())
}

/// Non-blocking drain of the chat streaming channel.
fn drain_chat_tokens(app: &mut App) {
    use tokio::sync::mpsc::error::TryRecvError;

    let mut finished = false;

    if let Some(rx) = &mut app.chat_rx {
        loop {
            match rx.try_recv() {
                Ok(ChatToken::AssistantStart) => {
                    app.chat_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: String::new(),
                    });
                    app.chat_scroll = u16::MAX / 2;
                }
                Ok(ChatToken::Token(tok)) => {
                    if let Some(msg) = app.chat_messages.last_mut() {
                        if msg.role == "assistant" {
                            msg.content.push_str(&tok);
                        }
                    }
                    app.chat_scroll = u16::MAX / 2;
                }
                Ok(ChatToken::ToolCall(display)) => {
                    app.chat_messages.push(ChatMessage {
                        role: "tool_call".into(),
                        content: display,
                    });
                    app.chat_scroll = u16::MAX / 2;
                }
                Ok(ChatToken::ToolResult(display)) => {
                    app.chat_messages.push(ChatMessage {
                        role: "tool_result".into(),
                        content: display,
                    });
                    app.chat_scroll = u16::MAX / 2;
                }
                Ok(ChatToken::Done) => {
                    finished = true;
                    break;
                }
                Ok(ChatToken::Error(e)) => {
                    // Append error to last assistant bubble or create one
                    match app.chat_messages.last_mut() {
                        Some(msg) if msg.role == "assistant" => {
                            msg.content = format!("[Error: {e}]");
                        }
                        _ => {
                            app.chat_messages.push(ChatMessage {
                                role: "assistant".into(),
                                content: format!("[Error: {e}]"),
                            });
                        }
                    }
                    finished = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }
    }

    if finished {
        app.chat_rx = None;
        app.chat_waiting = false;
    }
}

/// Background task: streams inference tokens (and tool calls) through `tx`.
///
/// If tools are enabled in config, the function uses non-streaming completion
/// to detect tool calls, executes them, and loops until the model stops calling
/// tools.  Regular streaming is used for the final (or only) response.
async fn do_chat_stream(
    messages: Vec<ChatMessage>,
    tx: tokio::sync::mpsc::UnboundedSender<ChatToken>,
    cfg: Config,
    registry: crate::registry::Registry,
) {
    use futures_util::StreamExt;

    let active = match registry.models.iter().find(|m| m.active) {
        Some(m) => m.clone(),
        None => {
            let _ = tx.send(ChatToken::Error(
                "No active model set. Run: deepsage switch <model>".into(),
            ));
            return;
        }
    };

    let client = reqwest::Client::new();
    let tools_enabled = &cfg.tools.enabled;

    // Build API messages from display messages (skip tool_call/tool_result display rows)
    let mut api_messages: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| !matches!(m.role.as_str(), "tool_call" | "tool_result"))
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();

    // ── Tool-calling loop (non-streaming) ─────────────────────────────────────
    if !tools_enabled.is_empty() {
        let tools_json = crate::tools::tools_json(tools_enabled);
        const MAX_ITERS: u32 = 6;
        let mut iters = 0u32;

        loop {
            if iters >= MAX_ITERS {
                let _ = tx.send(ChatToken::Error(
                    "Reached maximum tool-call iterations.".into(),
                ));
                return;
            }
            iters += 1;

            let resp_val =
                call_non_streaming(&client, &active, &cfg, &api_messages, Some(&tools_json)).await;

            let resp_val = match resp_val {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(ChatToken::Error(e.to_string()));
                    return;
                }
            };

            if crate::tools::is_tool_call_response(&resp_val) {
                let calls = crate::tools::extract_tool_calls(&resp_val);

                // Add assistant tool-call message to API conversation
                api_messages.push(resp_val["choices"][0]["message"].clone());
                // Ollama variant
                if api_messages.last().map(|v| v.is_null()).unwrap_or(true) {
                    api_messages.pop();
                    api_messages.push(resp_val["message"].clone());
                }

                for call in calls {
                    let display = format!("{}({})", call.name, summarise_args(&call.arguments));
                    let _ = tx.send(ChatToken::ToolCall(display));

                    let result = match crate::tools::execute(&call.name, &call.arguments).await {
                        Ok(s) => s,
                        Err(e) => format!("[error: {e}]"),
                    };
                    let _ = tx.send(ChatToken::ToolResult(truncate_display(&result, 120)));

                    // Tool result message for API
                    api_messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result,
                    }));
                }
                // Loop: ask model again with tool results
            } else {
                // No tool call — emit the content as streaming tokens then exit loop
                let content = crate::tools::extract_content(&resp_val);
                let _ = tx.send(ChatToken::AssistantStart);
                for chunk in content.chars().collect::<Vec<_>>().chunks(40) {
                    let s: String = chunk.iter().collect();
                    let _ = tx.send(ChatToken::Token(s));
                }
                let _ = tx.send(ChatToken::Done);
                return;
            }
        }
    }

    // ── Streaming path (no tools) ─────────────────────────────────────────────
    let _ = tx.send(ChatToken::AssistantStart);

    match active.backend.as_str() {
        "ollama" => {
            let tag = match &active.source {
                crate::registry::ModelSource::Ollama { tag } => tag.clone(),
                _ => active.name.clone(),
            };
            let url = format!("{}/api/chat", cfg.ollama.url.trim_end_matches('/'));
            let body =
                serde_json::json!({ "model": tag, "messages": api_messages, "stream": true });

            let resp = match client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(ChatToken::Error(format!("Ollama not reachable: {e}")));
                    return;
                }
            };

            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(_) => break,
                };
                for line in chunk.split(|&b| b == b'\n') {
                    if line.is_empty() {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                        let text = v["message"]["content"].as_str().unwrap_or("").to_string();
                        let done = v["done"].as_bool().unwrap_or(false);
                        if !text.is_empty() {
                            let _ = tx.send(ChatToken::Token(text));
                        }
                        if done {
                            break;
                        }
                    }
                }
            }
        }

        "llamacpp" => {
            let url = format!(
                "http://{}:{}/v1/chat/completions",
                cfg.llamacpp.host, cfg.llamacpp.port
            );
            let body = serde_json::json!({
                "model": active.name,
                "messages": api_messages,
                "stream": true,
            });

            let resp = match client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ =
                        tx.send(ChatToken::Error(format!("llama-server not reachable: {e}")));
                    return;
                }
            };

            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(_) => break,
                };
                for line in String::from_utf8_lossy(&chunk).lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            break;
                        }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            let text = v["choices"][0]["delta"]["content"]
                                .as_str()
                                .unwrap_or("")
                                .to_string();
                            if !text.is_empty() {
                                let _ = tx.send(ChatToken::Token(text));
                            }
                            if v["choices"][0]["finish_reason"].as_str() == Some("stop") {
                                break;
                            }
                        }
                    }
                }
            }
        }

        other => {
            let _ = tx.send(ChatToken::Error(format!("Unknown backend '{other}'")));
            return;
        }
    }

    let _ = tx.send(ChatToken::Done);
}

/// Send a non-streaming completion request (used for tool-calling detection).
async fn call_non_streaming(
    client: &reqwest::Client,
    model: &crate::registry::ModelEntry,
    cfg: &Config,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
) -> anyhow::Result<serde_json::Value> {
    match model.backend.as_str() {
        "ollama" => {
            let tag = match &model.source {
                crate::registry::ModelSource::Ollama { tag } => tag.clone(),
                _ => model.name.clone(),
            };
            let url = format!("{}/api/chat", cfg.ollama.url.trim_end_matches('/'));
            let mut body = serde_json::json!({
                "model": tag,
                "messages": messages,
                "stream": false,
            });
            if let Some(t) = tools {
                body["tools"] = serde_json::json!(t);
            }
            Ok(client.post(&url).json(&body).send().await?.json().await?)
        }
        "llamacpp" => {
            let url = format!(
                "http://{}:{}/v1/chat/completions",
                cfg.llamacpp.host, cfg.llamacpp.port
            );
            let mut body = serde_json::json!({
                "model": model.name,
                "messages": messages,
                "stream": false,
            });
            if let Some(t) = tools {
                body["tools"] = serde_json::json!(t);
            }
            Ok(client.post(&url).json(&body).send().await?.json().await?)
        }
        other => anyhow::bail!("unknown backend '{other}'"),
    }
}

fn summarise_args(args_json: &str) -> String {
    // Show the argument values in a compact form: key=val, key=val
    if let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(args_json)
    {
        let pairs: Vec<String> = obj
            .iter()
            .map(|(k, v)| {
                let val = match v {
                    serde_json::Value::String(s) => truncate_display(s, 40),
                    other => truncate_display(&other.to_string(), 40),
                };
                format!("{k}={val}")
            })
            .collect();
        pairs.join(", ")
    } else {
        truncate_display(args_json, 60)
    }
}

fn truncate_display(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        format!("{}…", chars[..max].iter().collect::<String>())
    }
}

async fn refresh_data(app: &mut App) {
    app.resource_stats = monitor::collect();

    // Reload registry from disk (may have changed via CLI)
    if let Ok(reg) = registry::load() {
        app.registry = reg;
    }

    // Fetch llmfit recommendations — optional, skip if not installed
    if crate::hardware::check(&app.config.llmfit_path) {
        if let Ok(recs) = crate::hardware::recommendations(5, &app.config.llmfit_path) {
            app.recommendations = recs;
        }
        if app.system_info.is_none() {
            if let Ok(info) = crate::hardware::system_info(&app.config.llmfit_path) {
                app.system_info = Some(info);
            }
        }
    }

    // Check DeepSage server status via state file + live health probe
    app.server_running = false;
    app.running_models.clear();

    if let Some(state) = crate::server::read_serve_state() {
        app.server_port = state.port;
        app.server_active_model = state.active_model.clone();

        let url = format!("http://127.0.0.1:{}/health", state.port);
        if let Ok(client) = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(800))
            .build()
        {
            if client
                .get(&url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                app.server_running = true;
                app.running_models.push(crate::backends::RunningModel {
                    name: state.active_model.clone(),
                    backend: "serve".into(),
                    pid: Some(state.pid),
                    vram_gb: 0.0,
                    ram_gb: 0.0,
                    endpoint: Some(format!("http://127.0.0.1:{}/v1", state.port)),
                });
            }
        }
    }

    // Also surface standalone llama-server processes from proc_registry
    for entry in crate::proc_registry::list_live() {
        // Skip if already covered by deepsage serve
        if app
            .running_models
            .iter()
            .any(|m| m.name == entry.model_name)
        {
            continue;
        }
        app.running_models.push(crate::backends::RunningModel {
            name: entry.model_name.clone(),
            backend: "llamacpp".into(),
            pid: Some(entry.pid),
            vram_gb: 0.0,
            ram_gb: 0.0,
            endpoint: Some(format!("http://127.0.0.1:{}/v1", entry.port)),
        });
        // Mark server_running true if any process is alive
        if !app.server_running {
            app.server_running = true;
        }
    }

    app.last_refresh = std::time::Instant::now();
}
