use anyhow::{bail, Result};

use crate::backends::llamacpp::LlamaCppBackend;
use crate::backends::ollama::OllamaBackend;
use crate::config::{self, Config};
use crate::download::{self, DownloadSource};
use crate::hardware;
use crate::monitor;
use crate::registry::{self, ModelEntry, ModelSource};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn print_hr() {
    println!("{}", "─".repeat(72));
}

fn fit_color_ansi(fit: &str) -> &'static str {
    match fit.to_lowercase().as_str() {
        s if s.contains("perfect") => "\x1b[32m",
        s if s.contains("good") => "\x1b[92m",
        s if s.contains("ok") => "\x1b[33m",
        _ => "\x1b[31m",
    }
}

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";

// ── recommend ────────────────────────────────────────────────────────────────

pub async fn recommend(n: usize, cfg: &Config) -> Result<()> {
    if !hardware::check(&cfg.llmfit_path) {
        println!("{}", hardware::INSTALL_HINT);
        return Ok(());
    }
    let recs = hardware::recommendations(n, &cfg.llmfit_path)?;
    println!("{BOLD}{CYAN} llmfit Recommendations{RESET}");
    print_hr();
    println!(
        "{BOLD}{:<3} {:<40} {:<9} {:<6} {:<7} {:<12} {:<8} Runtime{RESET}",
        "#", "Model", "Fit", "Score", "VRAM", "Quant", "TPS"
    );
    print_hr();
    for (i, r) in recs.iter().enumerate() {
        let fc = fit_color_ansi(&r.fit_level);
        // Trim long HuggingFace-style names: keep last two path segments
        let short_name = r.name.splitn(2, '/').last().unwrap_or(&r.name);
        println!(
            "{:<3} {BOLD}{:<40}{RESET} {fc}{:<9}{RESET} {:<6.1} {:<7.1} {:<12} {:<8.1} {}",
            i + 1,
            short_name,
            r.fit_level,
            r.score,
            r.vram_required_gb,
            r.quantization,
            r.estimated_tps,
            r.backend
        );
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
    println!("  CPU       : {} ({} cores)", info.cpu_name, info.cpu_cores);
    println!("  RAM       : {:.1} GB", info.ram_gb);
    let gpu = if info.gpu_name.is_empty() {
        "none".into()
    } else {
        info.gpu_name.clone()
    };
    let mem_note = if info.unified_memory {
        " (unified)"
    } else {
        ""
    };
    println!("  GPU       : {}{}", gpu, mem_note);
    println!("  VRAM      : {:.1} GB{}", info.vram_gb, mem_note);
    Ok(())
}

// ── list ─────────────────────────────────────────────────────────────────────

pub async fn list(_running_only: bool, _cfg: &Config) -> Result<()> {
    let reg = registry::load().unwrap_or_default();

    println!("{BOLD}{CYAN} Registered Models{RESET}");
    print_hr();
    if reg.models.is_empty() {
        println!("{DIM}  No models yet.  Run: deepsage pick{RESET}");
    } else {
        println!(
            "{BOLD}{:<22} {:<10} {:<10} {:<6}{RESET}",
            "Name", "Backend", "VRAM", "Active"
        );
        print_hr();
        for m in &reg.models {
            let active = if m.active {
                "\x1b[32m●\x1b[0m active"
            } else {
                "○"
            };
            let alloc = if m.alloc_auto {
                "auto".into()
            } else {
                format!("{:.1}G", m.vram_alloc_gb)
            };
            println!(
                "{BOLD}{:<22}{RESET} {:<10} {:<10} {}",
                m.name, m.backend, alloc, active
            );
        }
    }
    println!();

    // Check inference server via state file + live health probe
    println!("{BOLD}{CYAN} Inference Server{RESET}");
    print_hr();
    match crate::server::read_serve_state() {
        Some(state) => {
            let url = format!("http://127.0.0.1:{}/health", state.port);
            let alive = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(1))
                .build()
                .ok()
                .map(|c| async move {
                    c.get(&url)
                        .send()
                        .await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false)
                });
            let alive = if let Some(f) = alive { f.await } else { false };

            if alive {
                println!(
                    "  \x1b[32m●\x1b[0m {BOLD}Running{RESET}  model: {BOLD}{}{RESET}",
                    state.active_model
                );
                println!(
                    "  {DIM}Endpoint : http://127.0.0.1:{}/v1/chat/completions{RESET}",
                    state.port
                );
                println!();
                println!("{DIM}  Python:{RESET}");
                println!("    from openai import OpenAI");
                println!(
                    "    client = OpenAI(base_url=\"http://127.0.0.1:{}/v1\", api_key=\"none\")",
                    state.port
                );
                println!("    r = client.chat.completions.create(model=\"{}\", messages=[{{\"role\":\"user\",\"content\":\"Hello\"}}])", state.active_model);
                println!("    print(r.choices[0].message.content)");
            } else {
                println!("  {DIM}○ State file exists but server not responding — may have crashed{RESET}");
                println!("  {DIM}  Restart with: deepsage serve{RESET}");
            }
        }
        None => {
            println!("  {DIM}○ Not running{RESET}");
            println!("  {DIM}  Start with: deepsage serve{RESET}");
        }
    }
    Ok(())
}

// ── register ─────────────────────────────────────────────────────────────────

pub fn register(
    name: &str,
    source: &str,
    backend: Option<&str>,
    quantization: Option<&str>,
    _cfg: &Config,
) -> Result<()> {
    let mut reg = registry::load().unwrap_or_default();

    let (model_source, derived_backend) = parse_source_for_register(source)?;
    let backend = backend.unwrap_or(&derived_backend).to_string();

    let mut entry = ModelEntry::new(name, model_source, &backend);
    if let Some(q) = quantization {
        entry.quantization = q.to_string();
    }

    reg.register(entry);
    registry::save(&reg)?;
    println!("Registered {BOLD}{name}{RESET} via {backend}");
    Ok(())
}

fn parse_source_for_register(source: &str) -> Result<(ModelSource, String)> {
    if let Some(tag) = source.strip_prefix("ollama:") {
        return Ok((
            ModelSource::Ollama {
                tag: tag.to_string(),
            },
            "ollama".into(),
        ));
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
        return Ok((
            ModelSource::Local {
                path: source.to_string(),
            },
            "llamacpp".into(),
        ));
    }
    // Bare model name → assume Ollama tag
    Ok((
        ModelSource::Ollama {
            tag: source.to_string(),
        },
        "ollama".into(),
    ))
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
    let (v, r) = if auto {
        (None, None)
    } else {
        (vram_gb, ram_gb)
    };
    if !reg.set_alloc(id_or_name, v, r) {
        bail!("model not found: {id_or_name}");
    }
    registry::save(&reg)?;
    if auto {
        println!("Set allocation for {BOLD}{id_or_name}{RESET} to {DIM}auto{RESET}");
    } else {
        println!(
            "Set allocation for {BOLD}{id_or_name}{RESET}: vram={:.1}GB ram={:.1}GB",
            vram_gb.unwrap_or(0.0),
            ram_gb.unwrap_or(0.0)
        );
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
            println!(
                "Started {BOLD}{model}{RESET} on http://{}:{port}",
                cfg.llamacpp.host
            );
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
    println!(
        "  curl {}/api/generate -d '{{\"model\":\"{model}\",\"keep_alive\":0}}'",
        cfg.ollama.url
    );
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
    ollama
        .pull(model, |status, completed, total| {
            if status != last_status {
                last_status = status.clone();
            }
            let pct = total.map(|t| (completed * 100).checked_div(t).unwrap_or(0));
            match pct {
                Some(p) => print!("\r  {status:<40} {p:>3}%  "),
                None => print!("\r  {status:<40}       "),
            }
            use std::io::Write;
            let _ = std::io::stdout().flush();
        })
        .await?;
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
                let files =
                    download::hf_list_files(&repo, hf_cfg.token.as_deref(), &hf_cfg.endpoint)
                        .await?;
                if files.is_empty() {
                    bail!("no GGUF files found in {repo}");
                }
                println!("Available GGUF files:");
                for (i, f) in files.iter().enumerate() {
                    let size_str = f
                        .size
                        .map(|s| format!("{:.1}GB", s as f64 / 1e9))
                        .unwrap_or_default();
                    println!("  [{i}] {} {DIM}{size_str}{RESET}", f.filename);
                }
                println!("Downloading first: {}", files[0].filename);
                files.into_iter().next().unwrap().filename
            };

            println!("Downloading {BOLD}{file}{RESET} from hf:{repo}…");
            let path = download::hf_download(
                &repo,
                &file,
                &dest_dir,
                hf_cfg.token.as_deref(),
                &hf_cfg.endpoint,
                |downloaded, total| {
                    let pct = total.map(|t| (downloaded * 100).checked_div(t).unwrap_or(0));
                    let mb = downloaded as f64 / 1e6;
                    match pct {
                        Some(p) => print!("\r  {mb:.1} MB  {p:>3}%  "),
                        None => print!("\r  {mb:.1} MB        "),
                    }
                    let _ = std::io::stdout().flush();
                },
            )
            .await?;
            println!("\nSaved to {}", path.display());
        }

        DownloadSource::DirectUrl { url, filename } => {
            println!("Downloading {BOLD}{filename}{RESET} from {DIM}{url}{RESET}…");
            let path = download::download_url(&url, &dest_dir, &filename, |downloaded, total| {
                let pct = total.map(|t| (downloaded * 100).checked_div(t).unwrap_or(0));
                let mb = downloaded as f64 / 1e6;
                match pct {
                    Some(p) => print!("\r  {mb:.1} MB  {p:>3}%  "),
                    None => print!("\r  {mb:.1} MB        "),
                }
                let _ = std::io::stdout().flush();
            })
            .await?;
            println!("\nSaved to {}", path.display());
        }
    }
    Ok(())
}

// ── pick ─────────────────────────────────────────────────────────────────────
// Select from llmfit recommendations, download GGUF from HuggingFace,
// register it with the llamacpp backend — no Ollama required.

pub async fn pick(n: usize, index: Option<usize>, cfg: &Config) -> Result<()> {
    use std::io::{self, BufRead, Write};

    // Verify llama-server is available — offer to install it automatically
    if crate::backends::llamacpp::resolve_binary(&cfg.llamacpp.server_binary).is_none() {
        println!("{BOLD}llama-server not found.{RESET}");
        println!("It is needed to run models locally.\n");
        if which::which("brew").is_ok() {
            print!("Install now via Homebrew? {BOLD}[Y/n]{RESET}: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().lock().read_line(&mut input)?;
            if input.trim().is_empty() || input.trim().to_lowercase() == "y" {
                println!("Running: brew install llama.cpp …");
                let status = std::process::Command::new("brew")
                    .args(["install", "llama.cpp"])
                    .status()?;
                if !status.success() {
                    bail!("brew install failed. Try manually: brew install llama.cpp");
                }
                println!("{BOLD}llama-server installed.{RESET}\n");
            } else {
                bail!("llama-server required. Install with:\n  brew install llama.cpp");
            }
        } else {
            bail!(
                "llama-server not found.\n\
                 macOS : brew install llama.cpp\n\
                 Linux : see https://github.com/ggerganov/llama.cpp/releases"
            );
        }
    }

    // Curated fallback list for when llmfit is not installed
    const CURATED: &[(&str, &str)] = &[
        ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
        ("DeepSeek-R1-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
        ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct"),
        ("Mistral-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("Gemma-2-2B-IT", "google/gemma-2-2b-it"),
        ("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ];

    // Build display list from llmfit (if available) or curated fallback
    struct PickEntry {
        display: String,
        repo: String,
    }
    let entries: Vec<PickEntry> = if hardware::check(&cfg.llmfit_path) {
        let recs = hardware::recommendations(n.max(8), &cfg.llmfit_path)?;
        println!("{BOLD}{CYAN} Hardware-Aware Recommendations (llmfit){RESET}");
        print_hr();
        println!(
            "{BOLD}{:<3} {:<36} {:<9} {:<7}{RESET}",
            "#", "Model", "Fit", "VRAM"
        );
        print_hr();
        let entries: Vec<PickEntry> = recs
            .iter()
            .map(|r| {
                let short = r.name.splitn(2, '/').last().unwrap_or(&r.name).to_string();
                PickEntry {
                    display: short,
                    repo: r.name.clone(),
                }
            })
            .collect();
        for (i, (e, r)) in entries.iter().zip(recs.iter()).enumerate() {
            let fc = fit_color_ansi(&r.fit_level);
            println!(
                "{:<3} {BOLD}{:<36}{RESET} {fc}{:<9}{RESET} {:.1}G",
                i + 1,
                e.display,
                r.fit_level,
                r.vram_required_gb
            );
        }
        print_hr();
        entries
    } else {
        let show: Vec<_> = CURATED.iter().take(n.max(CURATED.len())).collect();
        println!("{BOLD}{CYAN} Curated Models{RESET}  {DIM}(install llmfit for hardware-aware picks){RESET}");
        print_hr();
        println!("{BOLD}{:<3} {:<36}{RESET}", "#", "Model");
        print_hr();
        let entries: Vec<PickEntry> = show
            .iter()
            .enumerate()
            .map(|(i, (name, repo))| {
                println!("{:<3} {BOLD}{:<36}{RESET}", i + 1, name);
                PickEntry {
                    display: name.to_string(),
                    repo: repo.to_string(),
                }
            })
            .collect();
        print_hr();
        entries
    };

    let chosen_idx = if let Some(i) = index {
        if i == 0 || i > entries.len() {
            bail!("index must be between 1 and {}", entries.len());
        }
        i - 1
    } else {
        print!("Enter number [1-{}]: ", entries.len());
        io::stdout().flush()?;
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line)?;
        let num: usize = line
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid input — enter a number"))?;
        if num == 0 || num > entries.len() {
            bail!("index must be between 1 and {}", entries.len());
        }
        num - 1
    };

    let chosen = &entries[chosen_idx];
    println!("\nSelected: {BOLD}{}{RESET}", chosen.display);

    // Find GGUF files — try original repo then common GGUF repo variants
    let hf_cfg = &cfg.huggingface;
    println!("Searching HuggingFace for GGUF files…");
    let (gguf_repo, files) =
        find_gguf_repo(&chosen.repo, hf_cfg.token.as_deref(), &hf_cfg.endpoint).await?;
    println!(
        "Found {} GGUF file(s) in {DIM}{gguf_repo}{RESET}",
        files.len()
    );

    // Show available quants and auto-select best
    for (i, f) in files.iter().enumerate() {
        let size_str = f
            .size
            .map(|s| format!("{:.1} GB", s as f64 / 1e9))
            .unwrap_or_default();
        println!("  [{i}] {} {DIM}{size_str}{RESET}", f.filename);
    }
    let chosen_file = pick_best_quant(&files);
    println!("Auto-selected: {BOLD}{}{RESET}", chosen_file.filename);

    // Download with progress
    let dest_dir = config::models_dir()?;
    let size_str = chosen_file
        .size
        .map(|s| format!(" ({:.1} GB)", s as f64 / 1e9))
        .unwrap_or_default();
    println!("Downloading{size_str}…");
    let short_name = chosen.display.clone();
    let local_path = download::hf_download(
        &gguf_repo,
        &chosen_file.filename,
        &dest_dir,
        hf_cfg.token.as_deref(),
        &hf_cfg.endpoint,
        |downloaded, total| {
            let pct = total.map(|t| (downloaded * 100).checked_div(t).unwrap_or(0));
            let mb = downloaded as f64 / 1e6;
            match pct {
                Some(p) => print!("\r  {mb:.1} MB  {p:>3}%  "),
                None => print!("\r  {mb:.1} MB        "),
            }
            let _ = std::io::stdout().flush();
        },
    )
    .await?;
    println!("\nSaved to {DIM}{}{RESET}", local_path.display());

    // Extract quantization label from filename
    let quant = chosen_file
        .filename
        .rsplit('-')
        .next()
        .and_then(|s| s.strip_suffix(".gguf"))
        .unwrap_or("Q4_K_M")
        .to_string();

    let mut reg = registry::load().unwrap_or_default();
    let no_active = !reg.models.iter().any(|m| m.active);
    let mut entry = ModelEntry::new(
        &short_name,
        ModelSource::HuggingFace {
            repo: gguf_repo,
            file: chosen_file.filename.clone(),
        },
        "llamacpp",
    );
    entry.local_path = Some(local_path.to_string_lossy().into_owned());
    entry.quantization = quant;
    entry.alloc_auto = true;
    reg.register(entry);
    if no_active {
        reg.switch(&short_name);
    }
    registry::save(&reg)?;

    println!("{BOLD}{CYAN}✓ Done!{RESET}");
    println!("  Model    : {BOLD}{short_name}{RESET}");
    println!("  File     : {DIM}{}{RESET}", local_path.display());
    if no_active {
        println!("  Status   : set as active model");
    }
    println!();
    println!("Next step: {BOLD}deepsage serve{RESET}");
    Ok(())
}

// Find a HuggingFace repo that contains GGUF files for the given model.
// Tries: original repo → {org}/{model}-GGUF → bartowski/{model}-GGUF
async fn find_gguf_repo(
    model_name: &str,
    token: Option<&str>,
    endpoint: &str,
) -> Result<(String, Vec<download::HfSibling>)> {
    // 1. Original repo
    if let Ok(files) = download::hf_list_files(model_name, token, endpoint).await {
        if !files.is_empty() {
            return Ok((model_name.to_string(), files));
        }
    }

    let parts: Vec<&str> = model_name.splitn(2, '/').collect();
    if parts.len() == 2 {
        let model_part = parts[1];
        for repo in [
            format!("{}/{}-GGUF", parts[0], model_part), // same org
            format!("bartowski/{model_part}-GGUF"),      // bartowski (most common)
            format!("TheBloke/{model_part}-GGUF"),       // TheBloke fallback
        ] {
            if let Ok(files) = download::hf_list_files(&repo, token, endpoint).await {
                if !files.is_empty() {
                    println!("  {DIM}(found GGUFs in {repo}){RESET}");
                    return Ok((repo, files));
                }
            }
        }
    }

    bail!(
        "No GGUF files found for '{model_name}'.\n\
         Specify one manually: deepsage download hf:<owner>/<repo>/<file>.gguf"
    )
}

// Pick the best quantization from a list of GGUF files.
// Preference order: Q4_K_M > Q5_K_M > Q4_K_S > Q4_0 > Q3_K_M > first
fn pick_best_quant(files: &[download::HfSibling]) -> &download::HfSibling {
    for pref in &["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S"] {
        if let Some(f) = files
            .iter()
            .find(|f| f.filename.to_uppercase().contains(pref))
        {
            return f;
        }
    }
    &files[0]
}

// ── search ────────────────────────────────────────────────────────────────────

pub async fn search(query: &str, cfg: &Config) -> Result<()> {
    if !hardware::check(&cfg.llmfit_path) {
        println!("{}", hardware::INSTALL_HINT);
        return Ok(());
    }
    // llmfit search doesn't support --json; stream its output directly
    hardware::run_passthrough(&cfg.llmfit_path, &["search", query])
}

// ── monitor (live CLI) ────────────────────────────────────────────────────────

pub async fn monitor(_cfg: &Config) -> Result<()> {
    use crossterm::{cursor, execute, terminal};
    use std::io::Write;

    execute!(std::io::stdout(), cursor::Hide)?;
    println!("DeepSage monitor  (Ctrl-C to quit)\n");

    loop {
        let stats = monitor::collect();
        execute!(
            std::io::stdout(),
            cursor::MoveToColumn(0),
            terminal::Clear(terminal::ClearType::FromCursorDown)
        )?;

        fn bar(ratio: f32, width: u16) -> String {
            let filled = (ratio * width as f32).round() as usize;
            let empty = width as usize - filled;
            format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
        }

        println!(
            "  CPU  {}  {:.1}%",
            bar(stats.cpu_pct / 100.0, 30),
            stats.cpu_pct
        );
        println!(
            "  RAM  {}  {:.1}/{:.1} GB",
            bar(stats.ram_pct(), 30),
            stats.ram_used_gb,
            stats.ram_total_gb
        );
        if stats.swap_total_gb > 0.0 {
            println!(
                "  Swap {}  {:.1}/{:.1} GB",
                bar(stats.swap_pct(), 30),
                stats.swap_used_gb,
                stats.swap_total_gb
            );
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
        println!(
            "  llamacpp.host    : {}:{}",
            cfg.llamacpp.host, cfg.llamacpp.port
        );
        println!(
            "  hf.token         : {}",
            cfg.huggingface
                .token
                .as_deref()
                .map(|_| "****")
                .unwrap_or("not set")
        );
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

// ── endpoint ─────────────────────────────────────────────────────────────────
// Print the inference endpoint for the active (or named) model and verify it
// is reachable. This is the URL callers can POST to directly.

pub async fn endpoint(model_override: Option<&str>, cfg: &Config) -> Result<()> {
    let reg = registry::load().unwrap_or_default();

    let model = match model_override {
        Some(name) => reg
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model not found: {name}"))?,
        None => reg.models.iter().find(|m| m.active).ok_or_else(|| {
            anyhow::anyhow!(
                "no active model set — run `deepsage switch <model>` or `deepsage pick`"
            )
        })?,
    };

    let (url, example) = match model.backend.as_str() {
        "ollama" => {
            let tag = match &model.source {
                ModelSource::Ollama { tag } => tag.clone(),
                _ => model.name.clone(),
            };
            let base = cfg.ollama.url.trim_end_matches('/');
            let url = format!("{base}/api/generate");
            let example = format!(
                r#"curl {url} \
  -H 'Content-Type: application/json' \
  -d '{{"model":"{tag}","prompt":"Hello","stream":false}}'"#
            );
            (url, example)
        }
        "llamacpp" => {
            let base = format!("http://{}:{}", cfg.llamacpp.host, cfg.llamacpp.port);
            let url = format!("{base}/v1/chat/completions");
            let example = format!(
                r#"curl {url} \
  -H 'Content-Type: application/json' \
  -d '{{"model":"{}","messages":[{{"role":"user","content":"Hello"}}]}}'  "#,
                model.name
            );
            (url, example)
        }
        other => bail!("unknown backend '{other}' for model '{}'", model.name),
    };

    // Check reachability
    let client = reqwest::Client::new();
    let reachable = client
        .get(
            url.trim_end_matches("/api/generate")
                .trim_end_matches("/v1/chat/completions"),
        )
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .is_ok();

    println!("{BOLD}{CYAN} Inference Endpoint{RESET}");
    print_hr();
    println!("  Model   : {BOLD}{}{RESET}", model.name);
    println!("  Backend : {}", model.backend);
    println!("  URL     : {BOLD}{CYAN}{url}{RESET}");
    println!(
        "  Status  : {}",
        if reachable {
            "\x1b[32m● reachable\x1b[0m"
        } else {
            "\x1b[31m● not reachable — start the backend first\x1b[0m"
        }
    );
    println!();
    println!("{DIM}Example request:{RESET}");
    println!("{example}");

    Ok(())
}

// ── infer ─────────────────────────────────────────────────────────────────────
// One-shot inference: send a prompt to the active model and print the response.

pub async fn infer(prompt: &str, model_override: Option<&str>, cfg: &Config) -> Result<()> {
    use std::io::Write;

    let reg = registry::load().unwrap_or_default();
    let model = match model_override {
        Some(name) => reg
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model not found: {name}"))?,
        None => reg.models.iter().find(|m| m.active).ok_or_else(|| {
            anyhow::anyhow!("no active model set — run `deepsage switch <model>` first")
        })?,
    };

    let client = reqwest::Client::new();

    match model.backend.as_str() {
        "ollama" => {
            let tag = match &model.source {
                ModelSource::Ollama { tag } => tag.clone(),
                _ => model.name.clone(),
            };
            let url = format!("{}/api/generate", cfg.ollama.url.trim_end_matches('/'));
            let body = serde_json::json!({
                "model": tag,
                "prompt": prompt,
                "stream": true,
            });
            let resp = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Ollama not reachable: {e}"))?;

            if !resp.status().is_success() {
                bail!(
                    "Ollama error {}: {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                );
            }

            use futures_util::StreamExt;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                for line in chunk.split(|&b| b == b'\n') {
                    if line.is_empty() {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                        if let Some(text) = v["response"].as_str() {
                            print!("{text}");
                            std::io::stdout().flush()?;
                        }
                        if v["done"].as_bool().unwrap_or(false) {
                            break;
                        }
                    }
                }
            }
            println!();
        }
        "llamacpp" => {
            let url = format!(
                "http://{}:{}/v1/chat/completions",
                cfg.llamacpp.host, cfg.llamacpp.port
            );
            let body = serde_json::json!({
                "model": model.name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": false,
            });
            let resp: serde_json::Value = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("llama-server not reachable: {e}"))?
                .json()
                .await?;
            let text = resp["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("[no response]");
            println!("{text}");
        }
        other => bail!("unknown backend '{other}'"),
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
