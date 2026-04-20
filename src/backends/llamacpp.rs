use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

use super::RunningModel;

type ProcessMap = Arc<Mutex<HashMap<String, Child>>>;

#[derive(Clone)]
pub struct LlamaCppBackend {
    pub server_binary: String,
    pub model_dir: PathBuf,
    pub host: String,
    pub base_port: u16,
    processes: ProcessMap,
}

impl LlamaCppBackend {
    pub fn new(server_binary: impl Into<String>, model_dir: PathBuf, host: impl Into<String>, base_port: u16) -> Self {
        Self {
            server_binary: server_binary.into(),
            model_dir,
            host: host.into(),
            base_port,
            processes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// List GGUF model files in the model directory.
    pub fn list_models(&self) -> Vec<PathBuf> {
        if !self.model_dir.exists() {
            return vec![];
        }
        std::fs::read_dir(&self.model_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Spawn llama-server for a model. Returns the port it is listening on.
    pub fn run(&self, model_path: &PathBuf, vram_gb: f32, ctx_size: u32) -> Result<u16> {
        let key = model_path.to_string_lossy().to_string();
        let mut processes = self.processes.lock().unwrap();

        if processes.contains_key(&key) {
            bail!("model is already running: {key}");
        }

        let port = self.next_free_port(&processes);
        let n_gpu_layers = if vram_gb > 0.0 { 999 } else { 0 };

        let child = Command::new(&self.server_binary)
            .args([
                "--model", &key,
                "--host", &self.host,
                "--port", &port.to_string(),
                "--n-gpu-layers", &n_gpu_layers.to_string(),
                "--ctx-size", &ctx_size.to_string(),
            ])
            .spawn()
            .with_context(|| format!("failed to spawn {}", self.server_binary))?;

        processes.insert(key, child);
        Ok(port)
    }

    pub fn stop(&self, model_path: &PathBuf) -> Result<()> {
        let key = model_path.to_string_lossy().to_string();
        let mut processes = self.processes.lock().unwrap();
        if let Some(mut child) = processes.remove(&key) {
            child.kill().context("failed to kill llama-server")?;
        } else {
            bail!("no running instance for: {key}");
        }
        Ok(())
    }

    pub fn stop_all(&self) {
        let mut processes = self.processes.lock().unwrap();
        for (_, mut child) in processes.drain() {
            let _ = child.kill();
        }
    }

    pub fn running_models(&self) -> Vec<RunningModel> {
        let processes = self.processes.lock().unwrap();
        processes
            .iter()
            .enumerate()
            .map(|(i, (path, child))| {
                let name = PathBuf::from(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(path)
                    .to_string();
                RunningModel {
                    name,
                    backend: "llamacpp".into(),
                    pid: Some(child.id()),
                    vram_gb: 0.0,
                    ram_gb: 0.0,
                    endpoint: Some(format!("http://{}:{}/v1", self.host, self.base_port + i as u16)),
                }
            })
            .collect()
    }

    fn next_free_port(&self, processes: &HashMap<String, Child>) -> u16 {
        self.base_port + processes.len() as u16
    }
}
