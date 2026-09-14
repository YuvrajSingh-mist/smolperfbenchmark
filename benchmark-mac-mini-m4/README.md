# Mac Mini M4 LLM Benchmark

Throughput and energy-efficiency benchmarks for dense, instruction-tuned LLMs on a Mac Mini M4 (16 GB unified memory). Measures tok/s, TTFT, ITL, and tok/J (tokens per joule) across 10 models and 15 prompt × gen combos per model using llama.cpp and Ollama backends.

---

## Table of Contents

- [Hardware](#hardware)
- [Models](#models)
- [Metrics](#metrics)
- [Installation & Setup](#installation--setup)
- [Running Benchmarks](#running-benchmarks)
  - [Arguments](#arguments)
- [Output](#output)

---

## Hardware

| | |
|---|---|
| Board | Mac Mini M4 (2025) |
| Chip | Apple M4 (10-core CPU, 10-core GPU, 16-core ANE) |
| Memory | 16 GB unified (CPU + GPU + ANE share the same pool) |
| Storage | NVMe SSD |
| OS | macOS Sequoia |

---

## Models

All models are dense (no MoE), Q4_K_M quantization, context 6144 tokens.

| Model | Family | Params | Quant | Est. Size | GGUF Source |
|---|---|---|---|---|---|
| Granite 4.1 3B | IBM Granite | 3B | Q4_K_M | ~2.0 GB | ibm-granite/granite-4.1-3b-GGUF (official) |
| Granite 4.1 8B | IBM Granite | 8B | Q4_K_M | ~5.0 GB | ibm-granite/granite-4.1-8b-GGUF (official) |
| Nemotron Mini 4B | NVIDIA Nemotron | 4B | Q4_K_M | ~2.7 GB | bartowski/Nemotron-Mini-4B-Instruct-GGUF |
| Nemotron Nano 8B | NVIDIA Nemotron | 8B | Q4_K_M | ~5.0 GB | bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF |
| Qwen3 4B | Alibaba Qwen3 | 4B | Q4_K_M | ~2.6 GB | Qwen/Qwen3-4B-GGUF (official) |
| Qwen3 8B | Alibaba Qwen3 | 8B | Q4_K_M | ~5.2 GB | Qwen/Qwen3-8B-GGUF (official) |
| Qwen2.5 7B | Alibaba Qwen2.5 | 7B | Q4_K_M | ~4.7 GB | Qwen/Qwen2.5-7B-Instruct-GGUF (official) |
| Gemma 3 4B | Google Gemma 3 | 4B | Q4_K_M | ~2.8 GB | ggml-org/gemma-3-4b-it-GGUF |
| Gemma 3 9B | Google Gemma 3 | 9B | Q4_K_M | ~5.8 GB | ggml-org/gemma-3-9b-it-GGUF |
| Gemma 3 12B | Google Gemma 3 | 12B | Q4_K_M | ~7.8 GB | ggml-org/gemma-3-12b-it-GGUF |

Models excluded: anything < 2B (not representative), MoE architectures (Granite 4.0 Tiny is MoE), Gemma 4 (smallest dense variant is 31B, ~17 GB at Q4_K_M — exceeds 16 GB).

---

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to first token (ms) — prefill latency |
| **ITL** | Inter-token latency (ms) — inverse of sustained decode speed |
| **Tok/s** | Output token throughput per user (p50) |
| **Prefill TPS** | Tokens processed during prefill per second |
| **Tok/J** | Tokens per joule — primary efficiency metric (see below) |
| **Peak RAM** | Peak RSS of inference process during combo run (MB) |

### Tok/J calculation

Power is sampled via macOS `powermetrics` (CPU + GPU + ANE combined) at 100 ms intervals. Each 500ms-combo window gets its own isolated `powermetrics.log`.

Per-request phase timestamps from `profile_export.jsonl` (`request_start_ns`, `request_ack_ns`, `request_end_ns`) are used to classify each power sample as prefill or decode:

```
tok/J = OSL_p50 / (decode_power_W × p50_decode_s)
```

Decode-only energy is used because prefill is a one-time prompt cost; decode is the sustained generation load that matters for efficiency comparisons.

### Sweep

| Prompt tokens | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| **Gen tokens** | 256, 512, 1024 | 256, 512, 1024 | 256, 512, 1024 | 256, 512, 1024 | 256, 512, 1024 |

15 combos per model × 10 models × 20 requests per combo. Context window fixed at 6144 tokens (max prompt 4096 + max gen 1024 + 1024 headroom).

---

## Installation & Setup

### Prerequisites

- Mac Mini M4 (or any Apple Silicon Mac with ≥ 16 GB)
- macOS Sequoia or later
- `sudo` access (required for `powermetrics`)
- Python 3.11+ (uv pins 3.12 via `.python-version`)
- `hf` CLI (comes with `uv sync` at the clone root)
- `aiperf` **0.11.0** in `.venv` (see step 5)

### 1. Install dependencies

Install Xcode command line tools (compiler, linker, Metal SDK):

```bash
xcode-select --install
```

Install CMake via Homebrew:

```bash
brew install cmake
```

### 2. Build llama.cpp with Metal

Clone and build from source. `-DGGML_METAL=ON` enables Apple GPU acceleration. `-DLLAMA_BUILD_EXAMPLES=ON` builds the actual executables (`llama-server`, `llama-cli`, `llama-bench`) on top of the core library — without it you only get the `.dylib`, no binaries.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
```

Binary will be at `~/llama.cpp/build/bin/llama-server`.

### 3. Install Ollama (optional, for Ollama backend)

```bash
brew install ollama
```

### 4. Clone this repo

Clone **anywhere** — Desktop is not required:

```bash
git clone https://github.com/YuvrajSingh-mist/smolperfbenchmark.git
cd smolperfbenchmark
```

### 5. Python env (`uv`)

All Python tools live in `.venv/` at the **clone root**. From that root:

```bash
brew install uv    # if needed; uv bundles its own Python (Homebrew CPython is broken on macOS 15)
uv sync --extra mac
.venv/bin/aiperf --version   # must print 0.11.0
```

That installs the locked **aiperf 0.11.0** (`44addf0`), `hf`, chart libs, and `mlx-lm`. Do not `pip install aiperf` from PyPI (yanked stub). Scripts find `.venv` automatically.

The benchmark script uses `hf download` to pull GGUFs on first run. Note: `huggingface_hub` ≥1.0 ships the CLI as `hf`, not `huggingface-cli`.

Tested on **MacBook Air M1 (2020)** and **Mac Mini M4 (2025), 16 GB**.

### 6. Authenticate with Hugging Face (required for gated models)

Some models used in this benchmark are **gated** on Hugging Face — you must accept the license on the model page while logged in, then authenticate this machine, before `aiperf` can load their tokenizer. Currently this applies to:

- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)

Without this, the benchmark aborts immediately with a `401 gated repo` error the first time it tries to run either model (see `abort()` behavior in [Running Benchmarks](#running-benchmarks) below — the script stops on the very first failed combo rather than silently skipping it).

Steps:

1. Visit each model page above while logged into your HF account and click "Acknowledge license" / "Agree and access repository".
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough).
3. Log in from this machine:

```bash
.venv/bin/hf auth login
# paste your token when prompted
```

This saves the token to `~/.cache/huggingface/token`, which `hf`, `aiperf`, and `transformers` all read automatically — no `--add-to-git-credential` or manual `HF_TOKEN` export needed for local use.

---

## Running Benchmarks

```bash
cd benchmark-mac-mini-m4/non-reasoning-models
bash benchmark_non_reasoning.sh [OPTIONS]
```

The script auto-relaunches inside tmux. Attach with `tmux attach -t non-reasoning-bench`. You can also start tmux yourself:

```bash
tmux new-session -d -s bench && \
tmux send-keys -t bench "cd /path/to/smolperfbenchmark/benchmark-mac-mini-m4/non-reasoning-models && bash benchmark_non_reasoning.sh" Enter && \
tmux attach -t bench
```

Detach: `Ctrl+B D` — reattach: `tmux attach -t bench`

**The script stops immediately on any failure** (bad download, server won't start, failed smoke test, gated-tokenizer 401, a single `aiperf` combo failing) rather than silently skipping the model and continuing the sweep — you'll see a `RUN STOPPED: <reason>` box the moment something breaks, instead of only finding out hours later when reading `report.md`. Fix the underlying issue, then re-run with `--resume <dir>` (see below) to continue exactly where it left off — already-completed combos are skipped automatically, so nothing needs to re-run.

Low disk space? Combine `--backend mlxlm` with `--stream` to download/benchmark/delete one model at a time instead of pulling the whole set upfront:

```bash
bash benchmark_non_reasoning.sh --backend mlxlm --stream
```

### Arguments

---

#### `--backend <llamacpp|mlxlm>`

- **Default:** `llamacpp`

Inference backend. `llamacpp` launches `llama-server` directly (GGUF, Q4_K_M). `mlxlm` launches `mlx_lm.server` against MLX 4-bit conversions of the same models instead — same aiperf sweep and methodology either way, so `report.md` rows are directly comparable across backends. Not every model has a published MLX 4-bit conversion; those are skipped automatically for `--backend mlxlm` (currently `nemotron-mini-4b` and `nemotron-nano-8b`).

```bash
bash benchmark_non_reasoning.sh --backend llamacpp   # GGUF via llama.cpp (default)
bash benchmark_non_reasoning.sh --backend mlxlm      # MLX 4-bit via mlx-lm
```

---

#### `--reqs <N>`

- **Default:** `20`

Number of requests per combo (prompt_len × gen_len cell).

---

#### `--only <model-name>`

- **Default:** all models

Run a single model by substring match. Example: `--only qwen3-8b`

---

#### `--resume <dir>`

Reuse an existing artifact directory, skipping combos that already have results.

```bash
bash benchmark_non_reasoning.sh \
  --resume artifacts/llamacpp/mac-mini-m4-llamacpp-YYYYMMDD-HHMM \
  --reqs 20
```

---

#### `--skip-download`

Skip the HuggingFace model download check. Use when all GGUFs are already present.

---

#### `--skip-smoke`

Skip the per-model smoke test (saves ~1 min per model, removes early-failure detection).

---

#### `--no-power`

Skip `powermetrics` entirely. No sudo required. Power and tok/J columns will be empty in the report.

---

#### `--stream`

Download one model, benchmark all of its combos, delete it, then move to the next model instead of downloading everything upfront. Keeps disk usage to roughly one model's size at a time instead of the full model set (GGUFs run 2-8 GB each, MLX conversions similar) — useful when disk space is tight. Only deletes what `--stream` itself downloaded this run; a model you already had on disk before starting is never touched.

```bash
bash benchmark_non_reasoning.sh --backend mlxlm --stream
```

---

#### `--dry-run`

Print the full model/combo list without executing anything.

---

## Output

Each run creates a timestamped directory:

```
artifacts/llamacpp/mac-mini-m4-llamacpp-YYYYMMDD-HHMM/
├── <model-name>/
│   └── gen<G>/
│       └── ctx<P>/
│           ├── profile_export_aiperf.json      # aiperf summary (TTFT, ITL, tok/s, …)
│           ├── profile_export.jsonl             # per-request nanosecond timestamps
│           ├── powermetrics.log                 # 100 ms power samples for this combo
│           ├── rss.log                          # 500 ms RSS samples (KB) for this combo
│           └── combo_info.json                  # model name, quant, ctx
└── report.md                                    # auto-generated after all models complete
```

`report.md` is written automatically at the end of the run. To regenerate it manually after a partial or resumed run, the report Python is embedded in the script and runs as the final step.
