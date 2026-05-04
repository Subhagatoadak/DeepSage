use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub fn project_dirs() -> Result<ProjectDirs> {
    ProjectDirs::from("", "", "deepsage")
        .ok_or_else(|| anyhow::anyhow!("cannot determine config directory"))
}

pub fn config_path() -> Result<PathBuf> {
    Ok(project_dirs()?.config_dir().join("config.toml"))
}

pub fn data_dir() -> Result<PathBuf> {
    Ok(project_dirs()?.data_dir().to_path_buf())
}

pub fn models_dir() -> Result<PathBuf> {
    Ok(data_dir()?.join("models"))
}

pub fn server_state_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("server.json"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub url: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:11434".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    pub server_binary: String,
    pub host: String,
    pub port: u16,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            server_binary: "llama-server".into(),
            host: "127.0.0.1".into(),
            port: 8080,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    pub token: Option<String>,
    pub endpoint: String,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            token: None,
            endpoint: "https://huggingface.co".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolsConfig {
    /// Names of built-in tools the LLM may call during chat.
    /// Valid values: "shell", "read_file", "web_fetch"
    #[serde(default)]
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default = "default_backend")]
    pub default_backend: String,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub llamacpp: LlamaCppConfig,
    #[serde(default)]
    pub huggingface: HuggingFaceConfig,
    #[serde(default = "default_llmfit")]
    pub llmfit_path: String,
    #[serde(default)]
    pub default_model: String,
    #[serde(default)]
    pub tools: ToolsConfig,
}

fn default_backend() -> String {
    "ollama".into()
}
fn default_llmfit() -> String {
    "llmfit".into()
}

pub fn load() -> Result<Config> {
    let path = config_path()?;
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        Ok(toml::from_str(&content)?)
    } else {
        Ok(Config::default())
    }
}

pub fn save(cfg: &Config) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, toml::to_string_pretty(cfg)?)?;
    Ok(())
}
