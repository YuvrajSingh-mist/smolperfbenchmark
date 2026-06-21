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
| Board | Mac Mini M4 |
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
- Python 3.10+
- `huggingface-hub` CLI (`pip install huggingface-hub`)
- `aiperf` installed in `~/venv`

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

```bash
git clone <this-repo>
cd benchmark-mac-mini-m4
```

### 5. Set up the shared venv

All Python tools (aiperf, huggingface_hub) live in one shared venv at `~/Desktop/smolbenchmark/venv/`:

```bash
cd ~/Desktop/smolbenchmark
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -U "huggingface_hub"
```

The benchmark script uses `hf download` to pull GGUFs on first run. Note: `huggingface_hub` ≥1.0 ships the CLI as `hf`, not `huggingface-cli`.

### 7. Install aiperf

The `aiperf` package on PyPI (`pip install aiperf`) is a yanked non-functional placeholder — do not use it. Install from the real source into the repo's shared venv:

```bash
cd ~/Desktop/smolbenchmark
source venv/bin/activate
pip install "aiperf @ git+https://github.com/ai-dynamo/aiperf.git"
```

Verify:

```bash
~/Desktop/smolbenchmark/venv/bin/aiperf --version
# should print 0.11.0 or later
```

---

## Running Benchmarks

```bash
bash benchmark_non_reasoning.sh [OPTIONS]
```

Run inside `tmux` for long sessions — the full sweep takes several hours:

```bash
tmux new-session -d -s bench && \
tmux send-keys -t bench "cd ~/Desktop/smolbenchmark/benchmark-mac-mini-m4 && \
bash benchmark_non_reasoning.sh" Enter && \
tmux attach -t bench
```

Detach: `Ctrl+B D` — reattach: `tmux attach -t bench`

### Arguments

---

#### `--backend <llamacpp|ollama>`

- **Default:** `llamacpp`

Inference backend. `llamacpp` launches `llama-server` directly. `ollama` uses the Ollama daemon's OpenAI-compatible endpoint.

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
