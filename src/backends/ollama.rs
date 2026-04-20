use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use super::RunningModel;

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub details: OllamaModelDetails,
}

impl OllamaModel {
    pub fn size_gb(&self) -> f32 {
        self.size as f32 / 1024.0 / 1024.0 / 1024.0
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OllamaModelDetails {
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
    #[serde(default)]
    pub family: String,
}

#[derive(Debug, Clone, Serialize)]
struct PullRequest {
    name: String,
    stream: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DeleteRequest {
    name: String,
}

pub struct OllamaBackend {
    client: reqwest::Client,
    pub url: String,
}

impl OllamaBackend {
    pub fn new(url: impl Into<String>) -> Self {
        Self { client: reqwest::Client::new(), url: url.into() }
    }

    pub async fn health(&self) -> bool {
        self.client.get(format!("{}/", self.url)).send().await.is_ok()
    }

    pub async fn list_models(&self) -> Result<Vec<OllamaModel>> {
        let resp: serde_json::Value = self
            .client
            .get(format!("{}/api/tags", self.url))
            .send()
            .await
            .context("ollama not reachable")?
            .json()
            .await?;
        Ok(serde_json::from_value(resp["models"].clone()).unwrap_or_default())
    }

    pub async fn running_models(&self) -> Result<Vec<RunningModel>> {
        let resp: serde_json::Value = self
            .client
            .get(format!("{}/api/ps", self.url))
            .send()
            .await
            .context("ollama not reachable")?
            .json()
            .await?;
        let models: Vec<serde_json::Value> =
            serde_json::from_value(resp["models"].clone()).unwrap_or_default();
        Ok(models
            .iter()
            .map(|m| RunningModel {
                name: m["name"].as_str().unwrap_or("").to_string(),
                backend: "ollama".into(),
                pid: None,
                vram_gb: m["size_vram"].as_u64().unwrap_or(0) as f32 / 1024.0 / 1024.0 / 1024.0,
                ram_gb: m["size"].as_u64().unwrap_or(0) as f32 / 1024.0 / 1024.0 / 1024.0,
                endpoint: Some(format!("{}/api/generate", self.url)),
            })
            .collect())
    }

    /// Pull a model with a streaming progress callback.
    /// `progress_cb(status_msg, completed_bytes, total_bytes_option)`
    pub async fn pull<F>(&self, model: &str, mut progress_cb: F) -> Result<()>
    where
        F: FnMut(String, u64, Option<u64>),
    {
        let resp = self
            .client
            .post(format!("{}/api/pull", self.url))
            .json(&PullRequest { name: model.to_string(), stream: true })
            .send()
            .await
            .context("ollama not reachable")?;

        if !resp.status().is_success() {
            bail!("ollama pull failed: {}", resp.status());
        }

        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for line in chunk.split(|&b| b == b'\n') {
                if line.is_empty() { continue; }
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                    let status = v["status"].as_str().unwrap_or("").to_string();
                    let completed = v["completed"].as_u64().unwrap_or(0);
                    let total = v["total"].as_u64();
                    progress_cb(status, completed, total);
                }
            }
        }
        Ok(())
    }

    pub async fn delete(&self, model: &str) -> Result<()> {
        let resp = self
            .client
            .delete(format!("{}/api/delete", self.url))
            .json(&DeleteRequest { name: model.to_string() })
            .send()
            .await?;
        if !resp.status().is_success() {
            bail!("ollama delete failed: {}", resp.status());
        }
        Ok(())
    }
}
