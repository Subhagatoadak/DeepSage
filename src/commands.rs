use anyhow::{bail, Result};

use crate::backends::ollama::OllamaBackend;
use crate::backends::llamacpp::LlamaCppBackend;
use crate::config::{self, Config};
use crate::download::{self, DownloadSource};
use crate::hardware;
use crate::monitor;
use crate::registry::{self, ModelEntry, ModelSource};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn print_hr() { println!("{}", "─".repeat(72)); }

fn fit_color_ansi(fit: &str) -> &'static str {
    match fit.to_lowercase().as_str() {
        s if s.contains("perfect") => "\x1b[32m",
        s if s.contains("good")    => "\x1b[92m",
        s if s.contains("ok")      => "\x1b[33m",
        _                           => "\x1b[31m",
    }
}

const RESET: &str = "\x1b[0m";
const BOLD:  &str = "\x1b[1m";
const DIM:   &str = "\x1b[2m";
const CYAN:  &str = "\x1b[36m";

// ── recommend ────────────────────────────────────────────────────────────────

pub async fn recommend(n: usize, cfg: &Config) -> Result<()> {
    if !hardware::check(&cfg.llmfit_path) {
        println!("{}", hardware::INSTALL_HINT);
        return Ok(());
    }
    let recs = hardware::recommendations(n, &cfg.llmfit_path)?;
    println!("{BOLD}{CYAN} llmfit Recommendations{RESET}");
    print_hr();
    println!("{BOLD}{:<3} {:<24} {:<10} {:<6} {:<8} {:<10} {}{RESET}",
        "#", "Model", "Fit", "Score", "VRAM", "Quant", "Backend");
    print_hr();
    for (i, r) in recs.iter().enumerate() {
        let fc = fit_color_ansi(&r.fit_level);
        println!("{:<3} {BOLD}{:<24}{RESET} {fc}{:<10}{RESET} {:<6.2} {:<8.1} {:<10} {}",
            i + 1, r.name, r.fit_level, r.score,
            r.vram_required_gb, r.quantization, r.backend);
    }
    Ok(())
}

// ── system ───────────────────────────────────────────────────────────────────

pub async fn system_info(cfg: &Config) -> Result<()> {
    if !hardware::check(&cfg.llmfit_path) {
        println!("{}", hardware::INSTALL_HINT);
        return Ok(());
    }
    let info = hardware::system_info(&cfg.llmfit_path)?;
    println!("{BOLD}{CYAN} Hardware Info{RESET}");
    print_hr();
    println!("  CPU Cores : {}", info.cpu_cores);
    println!("  RAM       : {:.1} GB", info.ram_gb);
    println!("  GPU       : {}", if info.gpu_name.is_empty() { "none".into() } else { info.gpu_name });
    println!("  VRAM      : {:.1} GB", info.vram_gb);
    Ok(())
}

// ── list ─────────────────────────────────────────────────────────────────────

pub async fn list(running_only: bool, cfg: &Config) -> Result<()> {
    let reg = registry::load().unwrap_or_default();

    let ollama = OllamaBackend::new(&cfg.ollama.url);

    if !running_only {
        println!("{BOLD}{CYAN} Registered Models{RESET}");
        print_hr();
        if reg.models.is_empty() {
            println!("{DIM}  No models registered. Use: deepsage register <model>{RESET}");
        } else {
            println!("{BOLD}{:<20} {:<10} {:<32} {:<8} {:<6}{RESET}",
                "Name", "Backend", "Source", "VRAM", "Active");
            print_hr();
            for m in &reg.models {
                let active = if m.active { "\x1b[32m●\x1b[0m" } else { "○" };
                let alloc = if m.alloc_auto {
                    "auto".into()
                } else {
                    format!("{:.1}G", m.vram_alloc_gb)
                };
                println!("{BOLD}{:<20}{RESET} {:<10} {DIM}{:<32}{RESET} {:<8} {}",
                    m.name, m.backend, m.source.to_string(), alloc, active);
            }
        }
        println!();
    }

    println!("{BOLD}{CYAN} Running Models (Ollama){RESET}");
    print_hr();
    match ollama.running_models().await {
        Ok(models) if models.is_empty() => println!("{DIM}  None running{RESET}"),
        Ok(models) => {
            for m in models {
                println!("  \x1b[32m●\x1b[0m {BOLD}{:<20}{RESET} vram:{:.1}GB  {}",
                    m.name, m.vram_gb, m.endpoint.unwrap_or_default());
            }
        }
        Err(e) => println!("{DIM}  Ollama not available: {e}{RESET}"),
    }
    Ok(())
}

// ── register ─────────────────────────────────────────────────────────────────

pub fn register(
    name: &str,
    source: &str,
    backend: Option<&str>,
    quantization: Option<&str>,
    cfg: &Config,
) -> Result<()> {
    let mut reg = registry::load().unwrap_or_default();

    let (model_source, derived_backend) = parse_source_for_register(source)?;
    let backend = backend.unwrap_or(&derived_backend).to_string();

    let mut entry = ModelEntry::new(name, model_source, &backend);
    if let Some(q) = quantization { entry.quantization = q.to_string(); }

    reg.register(entry);
    registry::save(&reg)?;
    println!("Registered {BOLD}{name}{RESET} via {backend}");
    Ok(())
}

fn parse_source_for_register(source: &str) -> Result<(ModelSource, String)> {
    if let Some(tag) = source.strip_prefix("ollama:") {
        return Ok((ModelSource::Ollama { tag: tag.to_string() }, "ollama".into()));
    }
    if let Some(rest) = source.strip_prefix("hf:") {
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        if parts.len() >= 3 {
            return Ok((
                ModelSource::HuggingFace {
                    repo: format!("{}/{}", parts[0], parts[1]),
                    file: parts[2].to_string(),
                },
                "llamacpp".into(),
            ));
        }
        bail!("hf source must be hf:owner/repo/file.gguf");
    }
    if source.starts_with('/') || source.starts_with("./") || source.starts_with("~/") {
        return Ok((ModelSource::Local { path: source.to_string() }, "llamacpp".into()));
    }
    // Bare model name → assume Ollama tag
    Ok((ModelSource::Ollama { tag: source.to_string() }, "ollama".into()))
}

// ── switch ───────────────────────────────────────────────────────────────────

pub fn switch(id_or_name: &str) -> Result<()> {
    let mut reg = registry::load().unwrap_or_default();
    if reg.get(id_or_name).is_none() {
        bail!("model not found: {id_or_name}  (use `deepsage list` to see registered models)");
    }
    reg.switch(id_or_name);
    registry::save(&reg)?;
    println!("Switched active model to {BOLD}{id_or_name}{RESET}");
    Ok(())
}

// ── alloc ─────────────────────────────────────────────────────────────────────

pub fn set_alloc(
    id_or_name: &str,
    vram_gb: Option<f32>,
    ram_gb: Option<f32>,
    auto: bool,
) -> Result<()> {
    let mut reg = registry::load().unwrap_or_default();
    let (v, r) = if auto { (None, None) } else { (vram_gb, ram_gb) };
    if !reg.set_alloc(id_or_name, v, r) {
        bail!("model not found: {id_or_name}");
    }
    registry::save(&reg)?;
    if auto {
        println!("Set allocation for {BOLD}{id_or_name}{RESET} to {DIM}auto{RESET}");
    } else {
        println!("Set allocation for {BOLD}{id_or_name}{RESET}: vram={:.1}GB ram={:.1}GB",
            vram_gb.unwrap_or(0.0), ram_gb.unwrap_or(0.0));
    }
    Ok(())
}

// ── run ───────────────────────────────────────────────────────────────────────

pub async fn run_model(model: &str, backend_override: Option<&str>, cfg: &Config) -> Result<()> {
    let reg = registry::load().unwrap_or_default();
    let entry = reg.get(model);
    let backend = backend_override
        .or_else(|| entry.map(|e| e.backend.as_str()))
        .unwrap_or(&cfg.default_backend);

    match backend {
        "ollama" => {
            let ollama = OllamaBackend::new(&cfg.ollama.url);
            if !ollama.health().await {
                bail!("Ollama not running. Start it with: ollama serve");
            }
            println!("Starting {BOLD}{model}{RESET} via Ollama…");
            // ollama keeps models loaded — just generate a ping to load it
            println!("{DIM}Model will load on first inference request.{RESET}");
            println!("Endpoint: {}/api/generate", cfg.ollama.url);
        }
        "llamacpp" => {
            let model_path = if let Some(e) = entry {
                e.local_path.as_ref().map(std::path::PathBuf::from)
            } else {
                None
            };
            let Some(path) = model_path else {
                bail!("llama.cpp model has no local path. Download it first: deepsage download hf:owner/repo/file.gguf");
            };
            let backend = LlamaCppBackend::new(
                &cfg.llamacpp.server_binary,
                config::models_dir()?,
                &cfg.llamacpp.host,
                cfg.llamacpp.port,
            );
            let vram = entry.map(|e| e.vram_alloc_gb).unwrap_or(0.0);
            let port = backend.run(&path, vram, 4096)?;
            println!("Started {BOLD}{model}{RESET} on http://{}:{port}", cfg.llamacpp.host);
        }
        other => bail!("unknown backend: {other}  (use 'ollama' or 'llamacpp')"),
    }
    Ok(())
}

// ── stop ──────────────────────────────────────────────────────────────────────

pub async fn stop_model(model: &str, cfg: &Config) -> Result<()> {
    // Ollama unloads models after inactivity.
    // llama.cpp would require a persistent process registry (future work).
    println!("{DIM}Ollama unloads models after inactivity.{RESET}");
    println!("To force-unload, send a request with keep_alive=0:");
    println!("  curl {}/api/generate -d '{{\"model\":\"{model}\",\"keep_alive\":0}}'", cfg.ollama.url);
    Ok(())
}

// ── pull ─────────────────────────────────────────────────────────────────────

pub async fn pull_model(model: &str, cfg: &Config) -> Result<()> {
    let ollama = OllamaBackend::new(&cfg.ollama.url);
    if !ollama.health().await {
        bail!("Ollama not running. Start it with: ollama serve");
    }
    println!("Pulling {BOLD}{model}{RESET} via Ollama…");
    let mut last_status = String::new();
    ollama.pull(model, |status, completed, total| {
        if status != last_status {
            last_status = status.clone();
        }
        let pct = total.map(|t| if t > 0 { completed * 100 / t } else { 0 });
        match pct {
            Some(p) => print!("\r  {status:<40} {p:>3}%  "),
            None    => print!("\r  {status:<40}       "),
        }
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }).await?;
    println!("\nDone.");
    Ok(())
}

// ── download (HuggingFace / URL) ─────────────────────────────────────────────

pub async fn download(source: &str, file_override: Option<&str>, cfg: &Config) -> Result<()> {
    use std::io::Write;
    let dest_dir = config::models_dir()?;
    let hf_cfg = &cfg.huggingface;

    match download::parse_source(source)? {
        DownloadSource::HuggingFace { repo, file } => {
            let file = if let Some(f) = file_override.or(file.as_deref()) {
                f.to_string()
            } else {
                // List GGUF files and pick the first
                println!("Listing GGUF files in {BOLD}{repo}{RESET}…");
                let files = download::hf_list_files(
                    &repo,
                    hf_cfg.token.as_deref(),
                    &hf_cfg.endpoint,
                ).await?;
                if files.is_empty() {
                    bail!("no GGUF files found in {repo}");
                }
                println!("Available GGUF files:");
                for (i, f) in files.iter().enumerate() {
                    let size_str = f.size.map(|s| format!("{:.1}GB", s as f64 / 1e9)).unwrap_or_default();
                    println!("  [{i}] {} {DIM}{size_str}{RESET}", f.filename);
                }
                println!("Downloading first: {}", files[0].filename);
                files.into_iter().next().unwrap().filename
            };

            println!("Downloading {BOLD}{file}{RESET} from hf:{repo}…");
            let path = download::hf_download(
                &repo, &file, &dest_dir,
                hf_cfg.token.as_deref(),
                &hf_cfg.endpoint,
                |downloaded, total| {
                    let pct = total.map(|t| if t > 0 { downloaded * 100 / t } else { 0 });
                    let mb = downloaded as f64 / 1e6;
                    match pct {
                        Some(p) => print!("\r  {mb:.1} MB  {p:>3}%  "),
                        None    => print!("\r  {mb:.1} MB        "),
                    }
                    let _ = std::io::stdout().flush();
                },
            ).await?;
            println!("\nSaved to {}", path.display());
        }

        DownloadSource::DirectUrl { url, filename } => {
            println!("Downloading {BOLD}{filename}{RESET} from {DIM}{url}{RESET}…");
            let path = download::download_url(&url, &dest_dir, &filename, |downloaded, total| {
                let pct = total.map(|t| if t > 0 { downloaded * 100 / t } else { 0 });
                let mb = downloaded as f64 / 1e6;
                match pct {
                    Some(p) => print!("\r  {mb:.1} MB  {p:>3}%  "),
                    None    => print!("\r  {mb:.1} MB        "),
                }
                let _ = std::io::stdout().flush();
            }).await?;
            println!("\nSaved to {}", path.display());
        }
    }
    Ok(())
}

// ── search ────────────────────────────────────────────────────────────────────

pub async fn search(query: &str, cfg: &Config) -> Result<()> {
    if !hardware::check(&cfg.llmfit_path) {
        println!("{}", hardware::INSTALL_HINT);
        return Ok(());
    }
    let results = hardware::search(query, &cfg.llmfit_path)?;
    println!("{BOLD}{CYAN} Search: \"{query}\"{RESET}");
    print_hr();
    if results.is_empty() {
        println!("{DIM}  No results.{RESET}");
        return Ok(());
    }
    println!("{BOLD}{:<24} {:<10} {:<6} {:<8} {:<10}{RESET}",
        "Model", "Fit", "Score", "VRAM", "Quant");
    print_hr();
    for r in &results {
        let fc = fit_color_ansi(&r.fit_level);
        println!("{BOLD}{:<24}{RESET} {fc}{:<10}{RESET} {:<6.2} {:<8.1} {:<10}",
            r.name, r.fit_level, r.score, r.vram_required_gb, r.quantization);
    }
    Ok(())
}

// ── monitor (live CLI) ────────────────────────────────────────────────────────

pub async fn monitor(_cfg: &Config) -> Result<()> {
    use crossterm::{cursor, execute, terminal};
    use std::io::Write;

    execute!(std::io::stdout(), cursor::Hide)?;
    println!("DeepSage monitor  (Ctrl-C to quit)\n");

    loop {
        let stats = monitor::collect();
        execute!(std::io::stdout(), cursor::MoveToColumn(0), terminal::Clear(terminal::ClearType::FromCursorDown))?;

        fn bar(ratio: f32, width: u16) -> String {
            let filled = (ratio * width as f32).round() as usize;
            let empty  = width as usize - filled;
            format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
        }

        println!("  CPU  {}  {:.1}%", bar(stats.cpu_pct / 100.0, 30), stats.cpu_pct);
        println!("  RAM  {}  {:.1}/{:.1} GB", bar(stats.ram_pct(), 30), stats.ram_used_gb, stats.ram_total_gb);
        if stats.swap_total_gb > 0.0 {
            println!("  Swap {}  {:.1}/{:.1} GB", bar(stats.swap_pct(), 30), stats.swap_used_gb, stats.swap_total_gb);
        }
        std::io::stdout().flush()?;

        // Move cursor up to overwrite on next tick
        let lines: u16 = if stats.swap_total_gb > 0.0 { 3 } else { 2 };
        execute!(std::io::stdout(), cursor::MoveToPreviousLine(lines))?;

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

// ── configure ────────────────────────────────────────────────────────────────

pub fn configure(
    set_backend: Option<String>,
    set_ollama_url: Option<String>,
    set_model_dir: Option<String>,
    set_hf_token: Option<String>,
    mut cfg: Config,
) -> Result<()> {
    let mut changed = false;

    if let Some(b) = set_backend {
        cfg.default_backend = b;
        changed = true;
    }
    if let Some(u) = set_ollama_url {
        cfg.ollama.url = u;
        changed = true;
    }
    if let Some(_dir) = set_model_dir {
        // model_dir is derived from data_dir; stored separately not needed
        println!("{DIM}Model dir is managed automatically in the data directory.{RESET}");
    }
    if let Some(tok) = set_hf_token {
        cfg.huggingface.token = Some(tok);
        changed = true;
    }

    if changed {
        config::save(&cfg)?;
        println!("Configuration saved.");
    } else {
        // Print current config
        println!("{BOLD}{CYAN} DeepSage Configuration{RESET}");
        print_hr();
        println!("  default_backend  : {}", cfg.default_backend);
        println!("  ollama.url       : {}", cfg.ollama.url);
        println!("  llamacpp.binary  : {}", cfg.llamacpp.server_binary);
        println!("  llamacpp.host    : {}:{}", cfg.llamacpp.host, cfg.llamacpp.port);
        println!("  hf.token         : {}", cfg.huggingface.token.as_deref().map(|_| "****").unwrap_or("not set"));
        println!("  llmfit_path      : {}", cfg.llmfit_path);
        if let Ok(p) = config::config_path() {
            println!("\n  Config file: {DIM}{}{RESET}", p.display());
        }
        if let Ok(p) = config::models_dir() {
            println!("  Models dir:  {DIM}{}{RESET}", p.display());
        }
    }
    Ok(())
}

// ── delete ────────────────────────────────────────────────────────────────────

pub fn delete(id_or_name: &str) -> Result<()> {
    let mut reg = registry::load().unwrap_or_default();
    if reg.remove(id_or_name) {
        registry::save(&reg)?;
        println!("Removed {BOLD}{id_or_name}{RESET} from registry.");
    } else {
        bail!("model not found: {id_or_name}");
    }
    Ok(())
}
