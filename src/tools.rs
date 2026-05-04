/// Built-in tools that the open-source LLM can invoke during chat.
///
/// Three tools are provided: shell (run local commands), read_file (read
/// local files), and web_fetch (fetch a URL).  The model receives their
/// schemas via the `tools` field in the chat API request and signals which
/// tool to call via `finish_reason = "tool_calls"` in its response.
use anyhow::Result;

// ── Tool definitions ──────────────────────────────────────────────────────────

pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
}

pub const ALL_TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "shell",
        description: "Run a shell command and return stdout/stderr (max 2000 chars). \
                      Use for file listing, running scripts, checking system state.",
    },
    ToolDef {
        name: "read_file",
        description: "Read a local file and return its contents (max 4000 chars). \
                      Use absolute or relative paths.",
    },
    ToolDef {
        name: "web_fetch",
        description: "Fetch a URL (HTML/text) and return the response body (max 4000 chars). \
                      Useful for documentation or plain-text API responses.",
    },
];

/// Build the OpenAI-format `tools` JSON array for the given enabled tool names.
pub fn tools_json(enabled: &[String]) -> Vec<serde_json::Value> {
    enabled
        .iter()
        .filter_map(|name| match name.as_str() {
            "shell" => Some(serde_json::json!({
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command and return its stdout/stderr output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            })),
            "read_file" => Some(serde_json::json!({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the local filesystem",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute or relative path to the file"
                            }
                        },
                        "required": ["path"]
                    }
                }
            })),
            "web_fetch" => Some(serde_json::json!({
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch the text content of a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch"
                            }
                        },
                        "required": ["url"]
                    }
                }
            })),
            _ => None,
        })
        .collect()
}

// ── Tool execution ────────────────────────────────────────────────────────────

pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON string of arguments (may be a string or object depending on backend)
    pub arguments: String,
}

/// Execute a tool by name with its raw JSON arguments string.
/// Returns the result as a string to feed back to the model.
pub async fn execute(name: &str, arguments: &str) -> Result<String> {
    // Arguments may arrive as a JSON string (OpenAI format) or JSON object (Ollama)
    let args: serde_json::Value =
        serde_json::from_str(arguments).unwrap_or(serde_json::Value::Object(Default::default()));

    match name {
        "shell" => {
            let cmd = args["command"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'command' argument"))?
                .to_string();
            Ok(run_shell(&cmd).await)
        }
        "read_file" => {
            let path = args["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'path' argument"))?;
            Ok(read_file(path))
        }
        "web_fetch" => {
            let url = args["url"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing 'url' argument"))?
                .to_string();
            Ok(web_fetch(&url).await)
        }
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

/// Extract tool calls from an OpenAI-format or Ollama-format completion response.
pub fn extract_tool_calls(response: &serde_json::Value) -> Vec<ToolCall> {
    // OpenAI / llama.cpp: choices[0].message.tool_calls
    if let Some(calls) = response["choices"][0]["message"]["tool_calls"].as_array() {
        return calls
            .iter()
            .map(|c| {
                let id = c["id"].as_str().unwrap_or("call_0").to_string();
                let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                // arguments is a JSON string in OpenAI format
                let arguments = match &c["function"]["arguments"] {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                ToolCall {
                    id,
                    name,
                    arguments,
                }
            })
            .collect();
    }

    // Ollama: message.tool_calls
    if let Some(calls) = response["message"]["tool_calls"].as_array() {
        return calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let id = format!("call_{i}");
                let name = c["function"]["name"].as_str().unwrap_or("").to_string();
                // Ollama arguments is a JSON object
                let arguments = c["function"]["arguments"].to_string();
                ToolCall {
                    id,
                    name,
                    arguments,
                }
            })
            .collect();
    }

    vec![]
}

/// Check whether the response signals that tool calls were made.
pub fn is_tool_call_response(response: &serde_json::Value) -> bool {
    // OpenAI / llama.cpp
    if response["choices"][0]["finish_reason"].as_str() == Some("tool_calls") {
        return true;
    }
    // Ollama
    if response["done_reason"].as_str() == Some("tool_calls") {
        return true;
    }
    // Fallback: tool_calls array is non-empty
    !extract_tool_calls(response).is_empty()
}

/// Extract plain text content from a non-streaming completion response.
pub fn extract_content(response: &serde_json::Value) -> String {
    // OpenAI / llama.cpp
    if let Some(s) = response["choices"][0]["message"]["content"].as_str() {
        return s.to_string();
    }
    // Ollama
    if let Some(s) = response["message"]["content"].as_str() {
        return s.to_string();
    }
    String::new()
}

// ── Individual tool implementations ──────────────────────────────────────────

async fn run_shell(command: &str) -> String {
    #[cfg(target_os = "windows")]
    let fut = tokio::process::Command::new("cmd")
        .args(["/C", command])
        .output();
    #[cfg(not(target_os = "windows"))]
    let fut = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .output();
    match fut.await {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let out = if stderr.is_empty() {
                stdout.into_owned()
            } else if stdout.is_empty() {
                format!("[stderr]: {}", stderr)
            } else {
                format!("{}\n[stderr]: {}", stdout, stderr)
            };
            truncate(&out, 2000)
        }
        Err(e) => format!("[shell error: {e}]"),
    }
}

fn read_file(path: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(content) => truncate(&content, 4000),
        Err(e) => format!("[read error: {e}]"),
    }
}

async fn web_fetch(url: &str) -> String {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("deepsage/0.1")
        .build()
    else {
        return "[web_fetch: failed to build client]".into();
    };
    match client.get(url).send().await {
        Ok(resp) => match resp.text().await {
            Ok(text) => truncate(&text, 4000),
            Err(e) => format!("[body error: {e}]"),
        },
        Err(e) => format!("[request error: {e}]"),
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        format!(
            "{}…[{} chars truncated]",
            chars[..max_chars].iter().collect::<String>(),
            chars.len() - max_chars
        )
    }
}
