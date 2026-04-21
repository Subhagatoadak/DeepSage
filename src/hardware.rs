#![allow(dead_code)]

use anyhow::{bail, Result};
use serde::Deserialize;
use std::process::Command;

pub const INSTALL_HINT: &str = "\
llmfit not found. Install it with:
  brew install llmfit        (macOS / Linux via Homebrew)
  scoop install llmfit       (Windows via Scoop)
  port install llmfit        (macOS via MacPorts)";

const BREW_CANDIDATES: &[&str] = &[
    "/opt/homebrew/bin/llmfit", // Apple Silicon
    "/usr/local/bin/llmfit",    // Intel Mac / Linux Homebrew
    "/home/linuxbrew/.linuxbrew/bin/llmfit",
];

/// Resolve the llmfit binary to an absolute path, trying the configured
/// name via PATH first, then falling back to known Homebrew locations.
pub fn resolve(configured: &str) -> Option<String> {
    if let Ok(p) = which::which(configured) {
        return Some(p.to_string_lossy().into_owned());
    }
    // Direct absolute path given and exists
    if std::path::Path::new(configured).is_absolute() && std::path::Path::new(configured).exists() {
        return Some(configured.to_string());
    }
    // Well-known Homebrew install locations
    for &candidate in BREW_CANDIDATES {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    None
}

pub fn check(llmfit_path: &str) -> bool {
    resolve(llmfit_path).is_some()
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SystemInfo {
    #[serde(default)]
    pub cpu_cores: u32,
    #[serde(default)]
    pub cpu_name: String,
    #[serde(alias = "total_ram_gb", default)]
    pub ram_gb: f32,
    #[serde(default)]
    pub gpu_name: String,
    #[serde(alias = "gpu_vram_gb", default)]
    pub vram_gb: f32,
    #[serde(default)]
    pub unified_memory: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelRecommendation {
    pub name: String,
    #[serde(default)]
    pub fit_level: String,
    #[serde(default)]
    pub score: f32,
    #[serde(alias = "memory_required_gb", default)]
    pub vram_required_gb: f32,
    #[serde(alias = "context_length", default)]
    pub context_window: u32,
    #[serde(alias = "best_quant", default)]
    pub quantization: String,
    #[serde(alias = "runtime", default)]
    pub backend: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub estimated_tps: f32,
    #[serde(default)]
    pub category: String,
}

fn run_json<T: for<'de> Deserialize<'de>>(llmfit_path: &str, args: &[&str]) -> Result<T> {
    let bin = resolve(llmfit_path).ok_or_else(|| anyhow::anyhow!("{}", INSTALL_HINT))?;
    let out = Command::new(&bin).args(args).arg("--json").output()?;
    if !out.status.success() {
        bail!("llmfit: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(serde_json::from_str(&text)?)
}

/// Extract models array from llmfit JSON: tries `v["models"]`, then root array.
fn extract_models(v: serde_json::Value) -> serde_json::Value {
    if let Some(arr) = v.get("models") {
        return arr.clone();
    }
    if v.is_array() {
        return v;
    }
    serde_json::Value::Array(vec![])
}

pub fn system_info(llmfit_path: &str) -> Result<SystemInfo> {
    let v: serde_json::Value = run_json(llmfit_path, &["system"])?;
    // Output is `{"system": {...}}` — extract the inner object
    let obj = v.get("system").cloned().unwrap_or(v);
    Ok(serde_json::from_value(obj)?)
}

pub fn recommendations(n: usize, llmfit_path: &str) -> Result<Vec<ModelRecommendation>> {
    let v: serde_json::Value = run_json(llmfit_path, &["recommend", "-n", &n.to_string()])?;
    Ok(serde_json::from_value(extract_models(v))?)
}

pub fn fit_scores(perfect_only: bool, llmfit_path: &str) -> Result<Vec<ModelRecommendation>> {
    let mut args = vec!["fit"];
    if perfect_only {
        args.push("--perfect");
    }
    let v: serde_json::Value = run_json(llmfit_path, &args)?;
    Ok(serde_json::from_value(extract_models(v))?)
}

/// Run a llmfit subcommand and stream its output directly to the terminal.
/// Used for commands like `search` that don't support --json.
pub fn run_passthrough(llmfit_path: &str, args: &[&str]) -> Result<()> {
    let bin = resolve(llmfit_path).ok_or_else(|| anyhow::anyhow!("{}", INSTALL_HINT))?;
    let status = Command::new(&bin).args(args).status()?;
    if !status.success() {
        bail!("llmfit exited with status {status}");
    }
    Ok(())
}

pub fn search(query: &str, llmfit_path: &str) -> Result<Vec<ModelRecommendation>> {
    let v: serde_json::Value = run_json(llmfit_path, &["search", query])?;
    Ok(serde_json::from_value(extract_models(v))?)
}

pub fn model_info(model: &str, llmfit_path: &str) -> Result<ModelRecommendation> {
    let v: serde_json::Value = run_json(llmfit_path, &["info", model])?;
    let obj = if v.is_array() { v[0].clone() } else { v };
    Ok(serde_json::from_value(obj)?)
}
