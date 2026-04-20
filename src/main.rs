use anyhow::Result;
use clap::{Parser, Subcommand};

mod backends;
mod commands;
mod config;
mod download;
mod hardware;
mod monitor;
mod registry;
mod tui;

#[derive(Parser)]
#[command(
    name    = "deepsage",
    version,
    about   = "Manage and run open source LLM models — powered by llmfit hardware analysis",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Launch the TUI dashboard (default when no subcommand is given)
    Tui,

    /// Show hardware-aware model recommendations from llmfit
    Recommend {
        /// Number of recommendations to display
        #[arg(short, long, default_value_t = 5)]
        n: usize,
    },

    /// List registered and running models
    List {
        /// Show only currently running models
        #[arg(short, long)]
        running: bool,
    },

    /// Register a model in the DeepSage registry
    ///
    /// Source formats:
    ///   ollama:llama3.2:3b         — Ollama model tag
    ///   hf:owner/repo/file.gguf   — HuggingFace GGUF file
    ///   /path/to/model.gguf        — local file
    Register {
        /// Display name for the model
        name: String,
        /// Source (see above)
        #[arg(short, long)]
        source: String,
        /// Backend: ollama or llamacpp (auto-detected from source if omitted)
        #[arg(short, long)]
        backend: Option<String>,
        /// Quantization label (e.g. Q4_K_M)
        #[arg(short, long)]
        quantization: Option<String>,
    },

    /// Switch the active model
    Switch {
        /// Model name or ID
        model: String,
    },

    /// Set memory allocation for a model
    Alloc {
        /// Model name or ID
        model: String,
        /// VRAM to allocate in GB (omit to auto)
        #[arg(long)]
        vram: Option<f32>,
        /// RAM to allocate in GB (omit to auto)
        #[arg(long)]
        ram: Option<f32>,
        /// Reset to automatic allocation
        #[arg(long)]
        auto: bool,
    },

    /// Run a registered model
    Run {
        /// Model name or ID
        model: String,
        /// Override backend: ollama or llamacpp
        #[arg(short, long)]
        backend: Option<String>,
    },

    /// Stop a running model
    Stop {
        /// Model name or "all"
        model: String,
    },

    /// Pull/download a model via Ollama
    Pull {
        /// Ollama model tag (e.g. llama3.2:3b)
        model: String,
    },

    /// Download a model from HuggingFace or a direct URL
    ///
    /// Source formats:
    ///   hf:owner/repo/file.gguf   — specific file
    ///   hf:owner/repo              — list and download first GGUF
    ///   https://example.com/m.gguf — direct URL
    Download {
        /// Source (see above)
        source: String,
        /// Override filename when using hf:owner/repo (without /file)
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Search the llmfit model database
    Search {
        query: String,
    },

    /// Show hardware information detected by llmfit
    System,

    /// Live CPU/RAM monitor in the terminal
    Monitor,

    /// Remove a model from the registry
    Delete {
        /// Model name or ID
        model: String,
    },

    /// Show or update DeepSage configuration
    Config {
        /// Set default backend: ollama or llamacpp
        #[arg(long)]
        set_backend: Option<String>,
        /// Set Ollama server URL
        #[arg(long)]
        set_ollama_url: Option<String>,
        /// Set HuggingFace API token
        #[arg(long)]
        set_hf_token: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = config::load()?;

    match cli.command {
        None | Some(Cmd::Tui)                   => tui::run(cfg).await,
        Some(Cmd::Recommend { n })               => commands::recommend(n, &cfg).await,
        Some(Cmd::List { running })              => commands::list(running, &cfg).await,
        Some(Cmd::Register { name, source, backend, quantization }) => {
            commands::register(&name, &source, backend.as_deref(), quantization.as_deref(), &cfg)
        }
        Some(Cmd::Switch { model })              => commands::switch(&model),
        Some(Cmd::Alloc { model, vram, ram, auto }) => {
            commands::set_alloc(&model, vram, ram, auto)
        }
        Some(Cmd::Run { model, backend })        => commands::run_model(&model, backend.as_deref(), &cfg).await,
        Some(Cmd::Stop { model })                => commands::stop_model(&model, &cfg).await,
        Some(Cmd::Pull { model })                => commands::pull_model(&model, &cfg).await,
        Some(Cmd::Download { source, file })     => commands::download(&source, file.as_deref(), &cfg).await,
        Some(Cmd::Search { query })              => commands::search(&query, &cfg).await,
        Some(Cmd::System)                        => commands::system_info(&cfg).await,
        Some(Cmd::Monitor)                       => commands::monitor(&cfg).await,
        Some(Cmd::Delete { model })              => commands::delete(&model),
        Some(Cmd::Config { set_backend, set_ollama_url, set_hf_token }) => {
            commands::configure(set_backend, set_ollama_url, None, set_hf_token, cfg)
        }
    }
}
