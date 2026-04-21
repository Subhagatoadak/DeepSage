
# DeepSage

Configure, run, manage, and monitor open source LLM models — with hardware-aware recommendations powered by [llmfit](https://github.com/AlexsJones/llmfit).
<img width="1842" height="1018" alt="Screenshot 2026-04-21 at 1 02 08 PM" src="https://github.com/user-attachments/assets/c4a54e74-42b1-489f-836c-f44031d55d71" />
## Features

- **Hardware-fit recommendations** — integrates llmfit to score models against your CPU, RAM, and VRAM
- **Model registry** — register models from Ollama, HuggingFace, or local files; switch between them instantly
- **Memory allocation control** — auto-allocate based on llmfit scores, or set VRAM/RAM limits per model
- **Direct downloads** — pull GGUF models from HuggingFace (`hf:owner/repo/file.gguf`) or any URL
- **Dual backend** — run models via Ollama or llama.cpp (`llama-server`)
- **TUI dashboard** — live ratatui interface with model browser, resource gauges, and recommendation table
- **Live monitor** — inline CPU and RAM bar charts updated every 500ms

## Installation

### Homebrew (macOS / Linux)

```bash
brew install llmfit        # hardware detection dependency
brew tap subhagatoadak/deepsage
brew install deepsage
```

### From source

```bash
brew install llmfit
cargo install --path .
```

## Quick start

```bash
deepsage system            # detect your hardware via llmfit
deepsage recommend         # see which models fit your machine
deepsage                   # launch the TUI dashboard
```

## Commands

### Model discovery

```bash
deepsage recommend              # top 5 hardware-fit models (from llmfit)
deepsage recommend -n 10        # show top 10
deepsage search phi             # search llmfit model database
deepsage system                 # show CPU / RAM / GPU / VRAM
```

### Registry — register and manage models

```bash
# Register from Ollama
deepsage register llama3.2 --source ollama:llama3.2:3b

# Register a HuggingFace GGUF
deepsage register phi4 --source hf:microsoft/Phi-4-mini-instruct-gguf/Phi-4-mini-instruct-Q4_K_M.gguf

# Register a local file
deepsage register my-model --source /path/to/model.gguf

# List all registered models
deepsage list

# Switch the active model
deepsage switch llama3.2

# Remove from registry
deepsage delete phi4
```

### Memory allocation

```bash
# Set manual VRAM limit for a model
deepsage alloc llama3.2 --vram 4.0

# Set both VRAM and RAM
deepsage alloc phi4 --vram 6.0 --ram 2.0

# Reset to automatic (llmfit-driven)
deepsage alloc llama3.2 --auto
```

### Downloading models

```bash
# Download a specific GGUF from HuggingFace
deepsage download hf:microsoft/Phi-4-mini-instruct-gguf/Phi-4-mini-instruct-Q4_K_M.gguf

# List available GGUFs in a repo and download the first
deepsage download hf:mistralai/Mistral-7B-Instruct-v0.3-GGUF

# Download from a direct URL
deepsage download https://example.com/model.gguf

# Pull via Ollama
deepsage pull llama3.2:3b
```

### Running models

```bash
deepsage run llama3.2               # use default backend (config)
deepsage run llama3.2 --backend ollama
deepsage run phi4 --backend llamacpp

deepsage stop llama3.2              # unload from Ollama

deepsage list --running             # show running instances only
```

### Monitoring

```bash
deepsage monitor                    # live CPU / RAM bars (Ctrl-C to quit)
```

### Configuration

```bash
deepsage config                           # show current config
deepsage config --set-backend llamacpp    # change default backend
deepsage config --set-ollama-url http://localhost:11434
deepsage config --set-hf-token hf_...    # HuggingFace token for gated models
```

## TUI dashboard

Run `deepsage` with no arguments to open the interactive dashboard.

```text
 DeepSage │ Apple M3 Max  RAM 64GB  VRAM 18GB
──────────────────────────────────────────────────────────
 Dashboard [1]   Models [2]   System [3]   Logs [4]
──────────────────────────────────────────────────────────

 ┌ Running Models ──────────────────┐ ┌ Resources ──────────────┐
 │ Model        Backend   VRAM      │ │ CPU  [████░░░░░░]  38%  │
 │ llama3.2:3b  Ollama    2.1 GB    │ │ RAM  [████████░░]  61%  │
 └──────────────────────────────────┘ │ Swap [░░░░░░░░░░]   0%  │
                                      └─────────────────────────┘
 ┌ llmfit Recommendations ──────────────────────────────────────┐
 │ #  Model              Fit       Score  VRAM    Quant         │
 │ 1  llama3.2:3b        Perfect   0.95   2.1GB   Q4_K_M        │
 │ 2  mistral:7b         Good      0.82   4.8GB   Q4_K_M        │
 │ 3  qwen2.5:14b        Good      0.78   9.2GB   Q4_K_M        │
 └──────────────────────────────────────────────────────────────┘

 q:Quit  Tab:Next  r:Run  s:Stop  p:Pull  d:Del  /:Search
```

**Key bindings:**

| Key | Action |
| --- | ------ |
| `Tab` / `Shift-Tab` | Cycle tabs |
| `1` `2` `3` `4` | Jump to tab |
| `j` / `k` or `↑` / `↓` | Navigate rows |
| `/` | Search models |
| `r` | Run selected model |
| `s` | Stop selected model |
| `p` | Pull selected model |
| `d` | Delete selected model |
| `PgUp` / `PgDn` | Scroll logs |
| `q` / `Esc` | Quit |

## Configuration file

Stored at `~/.config/deepsage/config.toml`:

```toml
default_backend = "ollama"

[ollama]
url = "http://localhost:11434"

[llamacpp]
server_binary = "llama-server"
host = "127.0.0.1"
port = 8080

[huggingface]
# token = "hf_..."   # uncomment for gated models
endpoint = "https://huggingface.co"
```

Model files are stored in `~/.local/share/deepsage/models/` (Linux) or `~/Library/Application Support/deepsage/models/` (macOS).

## Requirements

| Dependency | Purpose | Install |
| --- | --- | --- |
| [llmfit](https://github.com/AlexsJones/llmfit) | Hardware detection & model scoring | `brew install llmfit` |
| [Ollama](https://ollama.com) *(optional)* | Pull and serve models via API | [ollama.com](https://ollama.com) |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) *(optional)* | Run GGUF models locally | `brew install llama.cpp` |

## Building from source

```bash
git clone https://github.com/subhagatoadak/DeepSage
cd DeepSage
cargo build --release
./target/release/deepsage --version
```

Requires Rust 1.88+ (`rustup update stable`).

## Publishing a release

Push a version tag to trigger the GitHub Actions release workflow, which cross-compiles for macOS (arm64 + x86), Linux (arm64 + x86), and Windows, then uploads binaries to the GitHub release. Update the `sha256` values in [Formula/deepsage.rb](Formula/deepsage.rb) after the release to enable `brew install`.

```bash
git tag v0.1.0
git push origin v0.1.0
```

## License

MIT
