use anyhow::{bail, Result};
use serde::Deserialize;
use std::process::Command;

pub const INSTALL_HINT: &str = "\
llmfit not found. Install it with:
  brew install llmfit        (macOS / Linux via Homebrew)
  scoop install llmfit       (Windows via Scoop)
  port install llmfit        (macOS via MacPorts)";

pub fn check(llmfit_path: &str) -> bool {
    which::which(llmfit_path).is_ok()
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SystemInfo {
    #[serde(default)]
    pub cpu_cores: u32,
    #[serde(default)]
    pub ram_gb: f32,
    #[serde(default)]
    pub gpu_name: String,
    #[serde(default)]
    pub vram_gb: f32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelRecommendation {
    #[serde(alias = "model")]
    pub name: String,
    #[serde(alias = "fit", default)]
    pub fit_level: String,
    #[serde(alias = "composite_score", default)]
    pub score: f32,
    #[serde(alias = "memory_gb", default)]
    pub vram_required_gb: f32,
    #[serde(alias = "context", default)]
    pub context_window: u32,
    #[serde(alias = "quant", default)]
    pub quantization: String,
    #[serde(alias = "provider", default)]
    pub backend: String,
}

fn run_json<T: for<'de> Deserialize<'de>>(llmfit_path: &str, args: &[&str]) -> Result<T> {
    let out = Command::new(llmfit_path)
        .args(args)
        .arg("--json")
        .output()?;
    if !out.status.success() {
        bail!("llmfit: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(serde_json::from_str(&text)?)
}

pub fn system_info(llmfit_path: &str) -> Result<SystemInfo> {
    // llmfit may return an array or object
    let v: serde_json::Value = run_json(llmfit_path, &["system"])?;
    let obj = if v.is_array() { v[0].clone() } else { v };
    Ok(serde_json::from_value(obj)?)
}

pub fn recommendations(n: usize, llmfit_path: &str) -> Result<Vec<ModelRecommendation>> {
    let v: serde_json::Value = run_json(llmfit_path, &["recommend", "-n", &n.to_string()])?;
    let arr = if v.is_array() { v } else { v["recommendations"].clone() };
    Ok(serde_json::from_value(arr)?)
}

pub fn fit_scores(perfect_only: bool, llmfit_path: &str) -> Result<Vec<ModelRecommendation>> {
    let mut args = vec!["fit"];
    if perfect_only { args.push("--perfect"); }
    let v: serde_json::Value = run_json(llmfit_path, &args)?;
    let arr = if v.is_array() { v } else { v["models"].clone() };
    Ok(serde_json::from_value(arr)?)
}

pub fn search(query: &str, llmfit_path: &str) -> Result<Vec<ModelRecommendation>> {
    let v: serde_json::Value = run_json(llmfit_path, &["search", query])?;
    let arr = if v.is_array() { v } else { v["models"].clone() };
    Ok(serde_json::from_value(arr)?)
}

pub fn model_info(model: &str, llmfit_path: &str) -> Result<ModelRecommendation> {
    let v: serde_json::Value = run_json(llmfit_path, &["info", model])?;
    let obj = if v.is_array() { v[0].clone() } else { v };
    Ok(serde_json::from_value(obj)?)
}
