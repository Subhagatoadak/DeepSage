# DeepSage Tutorial — From Zero to Running LLM

Run open-source language models locally, serve them on an OpenAI-compatible endpoint, and query them from Python notebooks — with no Ollama, no cloud API, and no manual process management.

---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Prerequisites](#2-prerequisites)
3. [Install DeepSage](#3-install-deepsage)
4. [Step 1 — Pick and Download a Model](#4-step-1--pick-and-download-a-model)
5. [Step 2 — Start the Inference Server](#5-step-2--start-the-inference-server)
6. [Step 3 — Monitor with the TUI Dashboard](#6-step-3--monitor-with-the-tui-dashboard)
7. [Step 4 — Query from Python](#7-step-4--query-from-python)
8. [Step 5 — Use in a Jupyter Notebook](#8-step-5--use-in-a-jupyter-notebook)
9. [Managing Multiple Models](#9-managing-multiple-models)
10. [Configuration](#10-configuration)
11. [CLI Reference](#11-cli-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. How It Works

```
deepsage pick
    │
    ├─ checks llama-server is installed (offers brew install if missing)
    ├─ shows hardware-aware model list (llmfit) or curated fallback
    ├─ downloads .gguf file directly from HuggingFace
    └─ registers model in local registry

deepsage serve
    │
    ├─ reads active model from registry
    ├─ spawns llama-server internally (port 8080, managed automatically)
    ├─ waits for llama-server to load the model
    └─ starts OpenAI-compatible HTTP proxy on port 8888

Python / Notebook
    │
    └─ POST http://127.0.0.1:8888/v1/chat/completions
         (standard openai SDK or plain requests — no API key needed)
```

DeepSage owns the full lifecycle. You never run `llama-server` or manage ports manually.

---

## 2. Prerequisites

| Requirement | macOS | Linux |
|---|---|---|
| **Homebrew** | [brew.sh](https://brew.sh) | [docs.brew.sh/Homebrew-on-Linux](https://docs.brew.sh/Homebrew-on-Linux) |
| **Rust toolchain** | `brew install rust` | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **llama.cpp** | auto-installed by `deepsage pick` | auto-installed by `deepsage pick` |
| **llmfit** *(optional)* | `brew install llmfit` | `brew install llmfit` |

**llmfit** is optional. When installed it provides hardware-aware model recommendations scored against your specific CPU/GPU/RAM. Without it, `deepsage pick` falls back to a curated list of popular models.

**Disk space**: GGUF models range from ~1 GB (1–3B parameter models) to ~8 GB (7–9B models). Downloads go to `~/Library/Application Support/deepsage/models/` on macOS or `~/.local/share/deepsage/models/` on Linux.

---

## 3. Install DeepSage

### Option A — Homebrew (recommended)

```bash
brew tap subhagatoadak/deepsage
brew install deepsage
deepsage --version
```

### Option B — Build from source

```bash
git clone https://github.com/subhagatoadak/DeepSage
cd DeepSage
cargo build --release
cargo install --path .       # installs to ~/.cargo/bin/deepsage
deepsage --version
```

### Updating an existing install

```bash
cd DeepSage
git pull
cargo install --path . --force
deepsage --version
```

> **Important**: if you build from source and `deepsage` still shows old behaviour, run `cargo install --path . --force` — this replaces the binary in `~/.cargo/bin/`.

---

## 4. Step 1 — Pick and Download a Model

```bash
deepsage pick
```

This single command:

1. Checks that `llama-server` is installed. If not, it offers to run `brew install llama.cpp` for you.
2. Shows a ranked model list.
3. Downloads the selected model as a quantized GGUF file from HuggingFace.
4. Registers the model and sets it as active.

### 4.1 llama-server auto-install

If `llama-server` is not found on your PATH or in standard Homebrew locations, you will see:

```
llama-server not found.
It is needed to run models locally.

Install now via Homebrew? [Y/n]:
```

Press Enter (or type `y`) to install automatically. Alternatively install it yourself first:

```bash
brew install llama.cpp
```

### 4.2 Model selection

**With llmfit installed** — hardware-aware recommendations scored and ranked for your machine:

```
 Hardware-Aware Recommendations (llmfit)
────────────────────────────────────────────────────────────────────────
#   Model                                Fit       VRAM
────────────────────────────────────────────────────────────────────────
1   DeepSeek-R1-Distill-Qwen-7B          Perfect   5.2G
2   Qwen2.5-Coder-7B-Instruct            Perfect   5.2G
3   Llama-3.2-1B-Instruct                Perfect   1.8G
4   Qwen3-VL-4B-Instruct                 Perfect   1.9G
...
────────────────────────────────────────────────────────────────────────
Enter number [1-10]:
```

**Without llmfit** — curated list of well-tested models:

```
 Curated Models  (install llmfit for hardware-aware picks)
────────────────────────────────────────────────────────────────────────
#   Model
────────────────────────────────────────────────────────────────────────
1   Qwen2.5-7B-Instruct
2   DeepSeek-R1-7B
3   Llama-3.2-3B-Instruct
4   Phi-3.5-mini
5   Mistral-7B-Instruct
6   Qwen2.5-1.5B-Instruct
7   Gemma-2-2B-IT
8   SmolLM2-1.7B
────────────────────────────────────────────────────────────────────────
Enter number [1-8]:
```

> **Tip — start small**: models 1–2 GB (1–3B parameters) load in seconds and fit in 2 GB of RAM. 7B models need ~5 GB VRAM/RAM and take longer to load.

### 4.3 Download progress

After you enter a number, DeepSage searches HuggingFace for GGUF files. It automatically selects the best quantization (preferring Q4\_K\_M for the best quality/size balance) and downloads it.

```
Selected: Llama-3.2-1B-Instruct

Searching HuggingFace for GGUF files…
  (found GGUFs in bartowski/Llama-3.2-1B-Instruct-GGUF)
Found 8 GGUF file(s) in bartowski/Llama-3.2-1B-Instruct-GGUF
  [0] Llama-3.2-1B-Instruct-Q2_K.gguf 0.7 GB
  [1] Llama-3.2-1B-Instruct-Q4_K_M.gguf 0.8 GB
  [2] Llama-3.2-1B-Instruct-Q5_K_M.gguf 0.9 GB
  [3] Llama-3.2-1B-Instruct-Q8_0.gguf 1.3 GB
  ...
Auto-selected: Llama-3.2-1B-Instruct-Q4_K_M.gguf
Downloading (0.8 GB)…
  812.4 MB  100%

Saved to /Users/you/Library/Application Support/deepsage/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

✓ Done!
  Model    : Llama-3.2-1B-Instruct
  File     : .../models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  Status   : set as active model

Next step: deepsage serve
```

**Quantization guide:**

| Quantization | Size | Quality | Use when |
| --- | --- | --- | --- |
| Q2\_K | smallest | lowest | very limited RAM |
| Q4\_K\_M | **recommended** | good | default choice |
| Q5\_K\_M | medium | better | have extra RAM |
| Q8\_0 | large | near-lossless | want best quality |

### 4.4 Download a specific model manually

```bash
# Download GGUF directly
deepsage download hf:bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Register the downloaded file
deepsage register "Llama-3B" --source /path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Set as active
deepsage switch "Llama-3B"
```

---

## 5. Step 2 — Start the Inference Server

```bash
deepsage serve
```

### What happens internally

1. Reads the active model's local GGUF file path from the registry.
2. Spawns `llama-server` on an internal port (default `8080`). You never interact with this process directly.
3. Polls `http://127.0.0.1:8080/health` every 500 ms until the model is fully loaded (up to 120 s).
4. Starts DeepSage's own OpenAI-compatible HTTP proxy on `http://127.0.0.1:8888`.
5. Writes a server state file (`server.json`) so the TUI and `deepsage list` can detect that the server is running.
6. On `Ctrl-C`: gracefully shuts down the proxy, kills `llama-server`, and deletes the state file.

### Expected output

```
  Spawning llama-server for 'Llama-3.2-1B-Instruct' on internal port 8080…
  Waiting for llama-server to be ready ✓

 DeepSage Inference Server
────────────────────────────────────────────────────────────────
  Listening on  http://127.0.0.1:8888
  Active model  Llama-3.2-1B-Instruct

  OpenAI-compatible endpoints:
    GET  http://127.0.0.1:8888/v1/models
    POST http://127.0.0.1:8888/v1/chat/completions
    GET  http://127.0.0.1:8888/health

  Python (openai SDK):
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="none")
    resp = client.chat.completions.create(
        model="Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(resp.choices[0].message.content)

  Press Ctrl-C to stop.
────────────────────────────────────────────────────────────────
```

> **Keep this terminal open.** The server runs in the foreground. Use a second terminal for the TUI or Python queries.

### Custom host and port

```bash
deepsage serve --port 9000
deepsage serve --host 0.0.0.0 --port 8888    # expose on all interfaces (LAN access)
```

### Stop the server

Press `Ctrl-C` in the server terminal. DeepSage kills `llama-server` automatically — no orphan processes.

---

## 6. Step 3 — Monitor with the TUI Dashboard

Open a **second terminal** and run:

```bash
deepsage
```

The TUI dashboard refreshes every 3 seconds.

### Title bar

```
 DeepSage │ Apple M3 Pro  RAM 36GB  VRAM 36GB  │ Server ● http://127.0.0.1:8888/v1 [Llama-3.2-1B-Instruct]
```

| Indicator | Meaning |
|---|---|
| `●` green | Server is running and responding |
| `○` grey | Server is stopped |

### Tabs

| Tab | Key | Contents |
|---|---|---|
| **Dashboard** | `1` | Server status panel, CPU/RAM gauges, llmfit recommendations |
| **Models** | `2` | Registered models — name, backend, quantization, VRAM allocation, active flag |
| **System** | `3` | CPU, RAM, GPU/VRAM from llmfit hardware scan |
| **Logs** | `4` | Event log with timestamped entries |

### Dashboard — Server panel

When the server is running, the top-left panel shows:

```
 Server ● Running
──────────────────────────────────────────────────────────
Model             Backend     OpenAI Endpoint
──────────────────────────────────────────────────────────
Llama-3.2-1B      llamacpp    http://127.0.0.1:8888/v1/chat/completions
```

When stopped:

```
 Server ○ Stopped
── stopped —  run: deepsage serve
```

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Next / previous tab |
| `1` `2` `3` `4` | Jump to tab directly |
| `j` / `↓` | Next row in table |
| `k` / `↑` | Previous row in table |
| `/` | Enter search mode (Models tab) |
| `Esc` / `Enter` | Exit search mode |
| `PgDn` / `PgUp` | Scroll logs |
| `q` or `Esc` | Quit TUI |

---

## 7. Step 4 — Query from Python

Install the Python dependencies once:

```bash
pip install openai requests
```

### 7.1 Health check

```python
import requests

resp = requests.get("http://127.0.0.1:8888/health")
print(resp.json())
# {'status': 'ok', 'active_model': 'Llama-3.2-1B-Instruct'}
```

### 7.2 List available models

```python
resp = requests.get("http://127.0.0.1:8888/v1/models")
for m in resp.json()["data"]:
    print(m["id"], "—", m["owned_by"])
```

### 7.3 Chat completion — requests

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8888/v1/chat/completions",
    json={
        "model": "Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": False,
    }
)
print(resp.json()["choices"][0]["message"]["content"])
```

### 7.4 Chat completion — openai SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8888/v1",
    api_key="none",    # no API key required
)

response = client.chat.completions.create(
    model="Llama-3.2-1B-Instruct",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "Explain what a transformer model is in two sentences."},
    ],
)
print(response.choices[0].message.content)
```

### 7.5 Streaming response

```python
stream = client.chat.completions.create(
    model="Llama-3.2-1B-Instruct",
    messages=[{"role": "user", "content": "Count from 1 to 10, one number per line."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

### 7.6 Multi-turn conversation

```python
history = [{"role": "system", "content": "You are a helpful assistant."}]

def chat(message: str) -> str:
    history.append({"role": "user", "content": message})
    resp = client.chat.completions.create(model="Llama-3.2-1B-Instruct", messages=history)
    reply = resp.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("My name is Alice."))
print(chat("What is my name?"))
```

### 7.7 Finding your model name

The model name in API calls must exactly match the name shown in `deepsage list`. Run:

```bash
deepsage list
```

or query the server:

```python
resp = requests.get("http://127.0.0.1:8888/v1/models")
MODEL = resp.json()["data"][0]["id"]   # first registered model
print("Using model:", MODEL)
```

---

## 8. Step 5 — Use in a Jupyter Notebook

A ready-to-run test notebook is at `notebooks/deepsage_test.ipynb`.

### Setup

```bash
pip install openai requests jupyter
jupyter notebook notebooks/deepsage_test.ipynb
```

Start `deepsage serve` in a terminal **before** running cells.

### Notebook structure

| Cell | What it does |
|---|---|
| **Config** | Sets `BASE_URL = "http://127.0.0.1:8888"` — change port here if needed |
| **1 · Health check** | `GET /health` — confirms server is up and shows active model |
| **2 · List models** | `GET /v1/models` — sets `MODEL` variable for subsequent cells |
| **3 · Debug** | Prints full raw JSON — useful when responses are empty |
| **4 · Chat (requests)** | Non-streaming chat via `requests` |
| **5 · Chat (openai SDK)** | Non-streaming chat via `openai` SDK |
| **6 · Streaming (requests)** | Token-by-token stream parsed manually |
| **7 · Streaming (openai SDK)** | Token-by-token stream via `openai` SDK |
| **8 · Multi-turn** | Stateful conversation |
| **9 · Legacy completions** | `POST /v1/completions` endpoint |
| **10 · Batch inference** | Three parallel requests with `ThreadPoolExecutor` |

### Changing the port

If you started the server with `--port 9000`, update the first code cell:

```python
BASE_URL = "http://127.0.0.1:9000"
```

---

## 9. Managing Multiple Models

### See all registered models and server status

```bash
deepsage list
```

```
 Registered Models
────────────────────────────────────────────────────────────────────────
Name                     Backend    VRAM       Active
────────────────────────────────────────────────────────────────────────
Llama-3.2-1B-Instruct    llamacpp   auto       ● active
Qwen2.5-7B-Instruct      llamacpp   auto       ○

 Inference Server
────────────────────────────────────────────────────────────────────────
  ● Running  model: Llama-3.2-1B-Instruct
  Endpoint : http://127.0.0.1:8888/v1/chat/completions

  Python:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="none")
    r = client.chat.completions.create(model="Llama-3.2-1B-Instruct", ...)
    print(r.choices[0].message.content)
```

### Switch active model

Only one model can be active at a time. You must restart the server after switching.

```bash
deepsage switch "Qwen2.5-7B-Instruct"
# Restart the server (Ctrl-C the old one first):
deepsage serve
```

### Download and register a second model

```bash
deepsage pick          # pick another model — adds it alongside the first
deepsage switch "NewModel"
deepsage serve
```

### Register a local GGUF file

```bash
deepsage register "MyModel" --source /path/to/model.gguf
deepsage switch "MyModel"
deepsage serve
```

### Remove a model from the registry

```bash
deepsage delete "ModelName"
```

> This removes the registry entry only. The `.gguf` file on disk is **not** deleted — remove it manually if you want to free space.

### Memory allocation

By default allocation is `auto`. Override for specific models:

```bash
deepsage alloc "Qwen2.5-7B-Instruct" --vram 4.5 --ram 1.0
deepsage alloc "Qwen2.5-7B-Instruct" --auto     # back to auto
```

### One-shot inference from the terminal

Without starting the full server:

```bash
deepsage infer "Explain quantum entanglement in one sentence."
```

---

## 10. Configuration

Show current configuration:

```bash
deepsage config
```

```
 DeepSage Configuration
────────────────────────────────────────────────────────────────────────
  default_backend  : llamacpp
  ollama.url       : http://localhost:11434
  llamacpp.binary  : llama-server
  llamacpp.host    : 127.0.0.1:8080
  hf.token         : not set
  llmfit_path      : llmfit

  Config file: ~/Library/Application Support/deepsage/config.toml
  Models dir:  ~/Library/Application Support/deepsage/models
```

### Set a HuggingFace token

Required for gated models (Llama 3, Gemma) and for higher download rate limits:

```bash
deepsage config --set-hf-token hf_YourTokenHere
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### File locations

| File | Path (macOS) | Purpose |
| --- | --- | --- |
| Config | `~/Library/Application Support/deepsage/config.toml` | Settings |
| Registry | `~/Library/Application Support/deepsage/registry.json` | Model entries |
| Models | `~/Library/Application Support/deepsage/models/` | GGUF files |
| Server state | `~/Library/Application Support/deepsage/server.json` | Running server info (auto-managed) |

On Linux replace `~/Library/Application Support/` with `~/.local/share/`.

---

## 11. CLI Reference

```
USAGE
  deepsage [COMMAND] [OPTIONS]

COMMANDS
  (none)            Open TUI dashboard
  pick              Interactively pick, download, and register a model
  serve             Start OpenAI-compatible inference server
  list              Show registered models and server status
  switch <name>     Set active model
  register          Register a model from source string or file path
  delete <name>     Remove a model from the registry
  download          Download a GGUF from HuggingFace or a direct URL
  alloc             Set VRAM/RAM allocation for a model
  infer <prompt>    One-shot prompt to the active model (no server needed)
  endpoint          Show inference endpoint URL and test reachability
  recommend         Show llmfit hardware-aware recommendations
  search <query>    Search the llmfit model database
  system            Show hardware info (CPU, RAM, GPU, VRAM)
  monitor           Live CPU/RAM bar chart in the terminal
  config            Show or update configuration
  pull <tag>        Pull a model via Ollama (Ollama optional)
  run <model>       Start a model backend process
  stop <model>      Stop a running model process

SERVE OPTIONS
  --host <HOST>     Bind address    (default: 127.0.0.1)
  --port <PORT>     Listen port     (default: 8888)

PICK OPTIONS
  -n <N>            Number of recommendations to show   (default: 10)
  --index <N>       Auto-select entry N (skips interactive prompt)

REGISTER SOURCE FORMATS
  ollama:<tag>                  e.g. ollama:llama3.2:3b
  hf:<owner>/<repo>/<file>      e.g. hf:bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf
  /path/to/model.gguf           local file

API ENDPOINTS (when server is running)
  GET  /health                       server status + active model name
  GET  /v1/models                    list registered models
  POST /v1/chat/completions          chat inference (streaming + non-streaming)
  POST /v1/completions               legacy completion (maps to chat)
```

---

## 12. Troubleshooting

### `llama-server not found`

`deepsage pick` will offer to install it. Or install manually:

```bash
brew install llama.cpp
```

Verify:

```bash
which llama-server
llama-server --version
```

---

### `deepsage serve` fails: "model has no local GGUF file"

The active model in the registry was created without downloading its GGUF (e.g., an old Ollama-style entry). Fix:

```bash
deepsage list                         # identify the broken entry
deepsage delete "OldModelName"       # remove it
deepsage pick                         # download a fresh model
```

---

### `deepsage serve` fails: "llama-server did not become ready"

The model file may be corrupted, too large for available RAM, or the wrong architecture. Try:

```bash
# Check the file is readable
ls -lh "$(deepsage list | grep active | awk '{print $1}')"

# Try a smaller model
deepsage pick -n 10    # look for 1–3B models
```

---

### Chat completions return empty content

The server is running but returning `"content": ""`. This means `llama-server` received the request but produced no output. Check:

```bash
# Is the server up?
curl http://127.0.0.1:8888/health

# Raw response debug
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"YourModelName","messages":[{"role":"user","content":"Hi"}],"stream":false}' \
  | python3 -m json.tool
```

If `error` appears in the JSON, the model name in your request doesn't match the registered name. Run `deepsage list` to get the exact name.

---

### Download interrupted or stuck

HuggingFace downloads restart from the beginning if interrupted. For large files:

```bash
# Set a token for higher rate limits
deepsage config --set-hf-token hf_YourToken

# Or download with curl for resumable downloads
curl -L -C - -o model.gguf "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
# Then register the local file:
deepsage register "Llama-3B" --source /path/to/model.gguf
```

---

### Port already in use

```bash
# Use a different port
deepsage serve --port 9000

# Update Python client to match
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="none")

# Or find and kill whatever is using 8888
lsof -i :8888
kill <PID>
```

---

### TUI shows `○ not running` even though the server is running

The TUI reads `server.json` and then probes `/health`. Check directly:

```bash
curl http://127.0.0.1:8888/health
```

If this responds, the server is fine — the TUI refreshes every 3 seconds and will catch up. If the state file is stale from a previous crash, just start `deepsage serve` again — it overwrites the file on startup.

---

### Binary is outdated after `git pull`

```bash
cargo install --path . --force
deepsage --version
```

---

## Quick Reference Card

```bash
# ── First time setup ────────────────────────────────────────────────
brew install llmfit                  # optional, for hardware-aware picks

deepsage pick                        # installs llama.cpp if needed,
                                     # downloads model, registers it

# ── Every run ───────────────────────────────────────────────────────
deepsage serve                       # Terminal 1: start server
deepsage                             # Terminal 2: open TUI dashboard

# ── Check status ────────────────────────────────────────────────────
deepsage list                        # show models + server endpoint

# ── Python / notebook ───────────────────────────────────────────────
pip install openai

from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="none")
r = client.chat.completions.create(
    model="<name from deepsage list>",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(r.choices[0].message.content)

# ── Manage models ───────────────────────────────────────────────────
deepsage pick                        # add another model
deepsage switch "ModelName"          # change active model
deepsage delete "ModelName"          # remove from registry
deepsage list                        # always shows current state
```
