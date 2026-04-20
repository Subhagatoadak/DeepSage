use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

use crate::config::data_dir;

fn registry_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("registry.json"))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    Ollama { tag: String },
    HuggingFace { repo: String, file: String },
    Local { path: String },
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama { tag } => write!(f, "ollama:{tag}"),
            Self::HuggingFace { repo, file } => write!(f, "hf:{repo}/{file}"),
            Self::Local { path } => write!(f, "local:{path}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub source: ModelSource,
    pub backend: String,
    pub local_path: Option<String>,
    pub quantization: String,
    pub vram_alloc_gb: f32,
    pub ram_alloc_gb: f32,
    pub alloc_auto: bool,
    pub active: bool,
    pub registered_at: String,
}

impl ModelEntry {
    pub fn new(name: impl Into<String>, source: ModelSource, backend: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            source,
            backend: backend.into(),
            local_path: None,
            quantization: String::new(),
            vram_alloc_gb: 0.0,
            ram_alloc_gb: 0.0,
            alloc_auto: true,
            active: false,
            registered_at: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Registry {
    pub models: Vec<ModelEntry>,
    pub active_model: Option<String>,
}

impl Registry {
    pub fn get(&self, id_or_name: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id_or_name || m.name == id_or_name)
    }

    pub fn get_mut(&mut self, id_or_name: &str) -> Option<&mut ModelEntry> {
        self.models.iter_mut().find(|m| m.id == id_or_name || m.name == id_or_name)
    }

    pub fn register(&mut self, entry: ModelEntry) {
        // replace if same name already exists
        if let Some(existing) = self.models.iter_mut().find(|m| m.name == entry.name) {
            *existing = entry;
        } else {
            self.models.push(entry);
        }
    }

    pub fn remove(&mut self, id_or_name: &str) -> bool {
        let before = self.models.len();
        self.models.retain(|m| m.id != id_or_name && m.name != id_or_name);
        self.models.len() < before
    }

    /// Switch the active model. Returns the old active name if any.
    pub fn switch(&mut self, id_or_name: &str) -> Option<String> {
        let old = self.active_model.clone();
        for m in &mut self.models {
            m.active = m.id == id_or_name || m.name == id_or_name;
        }
        self.active_model = self.models.iter().find(|m| m.active).map(|m| m.name.clone());
        old
    }

    /// Set memory allocation for a model. Pass None for vram/ram to use auto.
    pub fn set_alloc(&mut self, id_or_name: &str, vram_gb: Option<f32>, ram_gb: Option<f32>) -> bool {
        if let Some(m) = self.get_mut(id_or_name) {
            match (vram_gb, ram_gb) {
                (None, None) => { m.alloc_auto = true; }
                (v, r) => {
                    m.alloc_auto = false;
                    if let Some(v) = v { m.vram_alloc_gb = v; }
                    if let Some(r) = r { m.ram_alloc_gb = r; }
                }
            }
            true
        } else {
            false
        }
    }

    /// Total manually allocated VRAM across all models.
    pub fn total_vram_alloc(&self) -> f32 {
        self.models.iter().filter(|m| !m.alloc_auto).map(|m| m.vram_alloc_gb).sum()
    }
}

pub fn load() -> Result<Registry> {
    let path = registry_path()?;
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&content)?)
    } else {
        Ok(Registry::default())
    }
}

pub fn save(reg: &Registry) -> Result<()> {
    let path = registry_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(reg)?)?;
    Ok(())
}
