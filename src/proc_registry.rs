use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcEntry {
    pub model_name: String,
    pub pid: u32,
    pub port: u16,
    pub model_path: String,
}

type ProcMap = HashMap<String, ProcEntry>;

fn proc_path() -> Result<std::path::PathBuf> {
    crate::config::data_dir().map(|d| d.join("procs.json"))
}

pub fn load() -> ProcMap {
    proc_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(map: &ProcMap) -> Result<()> {
    let path = proc_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(map)?)?;
    Ok(())
}

pub fn register_proc(entry: ProcEntry) -> Result<()> {
    let mut map = load();
    map.insert(entry.model_name.clone(), entry);
    save(&map)
}

pub fn remove_proc(model_name: &str) -> Result<Option<ProcEntry>> {
    let mut map = load();
    let removed = map.remove(model_name);
    save(&map)?;
    Ok(removed)
}

/// Return only entries whose process is still alive; lazily evict stale entries.
pub fn list_live() -> Vec<ProcEntry> {
    let map = load();
    let mut live = Vec::new();
    let mut stale = Vec::new();

    for (key, entry) in &map {
        if is_pid_alive(entry.pid) {
            live.push(entry.clone());
        } else {
            stale.push(key.clone());
        }
    }

    if !stale.is_empty() {
        let mut cleaned = map;
        for k in &stale {
            cleaned.remove(k);
        }
        let _ = save(&cleaned);
    }

    live
}

pub fn is_pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        // kill -0 checks existence without sending a signal
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}
