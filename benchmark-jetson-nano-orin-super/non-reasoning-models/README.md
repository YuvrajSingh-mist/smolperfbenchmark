# Non-Reasoning LLM Benchmark - Jetson Orin Nano Super 8GB

Throughput, latency, and energy-efficiency benchmarks for eight tiny instruct LLMs (135M–1.2B params) running on a **Jetson Orin Nano Super 8GB** at all four power modes.

Supports two backends from the same GGUF files — **llama.cpp** and **Ollama** — for an apples-to-apples comparison.

Key metric: **output tok/J** = OSL / (avg\_power\_W × RL\_p50\_s)

Published report: [benchmark_report.md](./benchmark_report.md)



## Table of Contents

- [Hardware](#hardware)
- [Models](#models)
- [Metrics](#metrics)
- [Prerequisites](#prerequisites)
- [Running Benchmarks](#running-benchmarks)
- [Output](#output)
- [Generating Charts](#generating-charts)
- [Arguments](#arguments)


## Report

To get a detailed report of the benchmark with full experiments and analysis, you can get it [here](https://www.smolhub.com/posts/jetson-nano-super-benchmark-non-reasoning/)`

## Hardware

| Component | Detail |
|-----------|--------|
| Board | Jetson Orin Nano Super 8GB Developer Kit |
| CPU | 6× Arm Cortex-A78AE @ up to 1.728 GHz |
| GPU | NVIDIA Ampere, 1024 CUDA cores, 32 Tensor cores |
| Memory | 8 GB LPDDR5 unified CPU + GPU |
| JetPack | R36.4.7 (Ubuntu 22.04, CUDA 12.6) |



## Models

| Model | Quant | GGUF size |
|-------|-------|----------:|
| SmolLM2-135M-Instruct | Q4\_K\_M | 101 MB |
| SmolLM2-360M-Instruct | Q8\_0 | 369 MB |
| Qwen2.5-0.5B-Instruct | Q4\_K\_M | 469 MB |
| LFM2.5-350M | Q4\_K\_M | 219 MB |
| LFM2.5-1.2B-Instruct | Q4\_K\_M | 698 MB |
| Qwen3-0.6B | Q8\_0 | 610 MB |
| Llama-3.2-1B-Instruct | Q4\_K\_M | 771 MB |
| Gemma3-1B-IT | Q4\_K\_M | 769 MB |
| Gemma3-4B-IT *(OOM at all modes)* | Q4\_K\_M | 2.4 GB |

GGUFs are auto-downloaded from Hugging Face on first run into `~/gguf-models/`.



## Metrics

| Metric | Description |
|--------|-------------|
| **output tok/J** | OSL / (avg\_power\_W × RL\_p50\_s) — primary efficiency metric |
| **TTFT p50** | Time to first token, median over 20 requests (ms) |
| **ITL p50** | Inter-token latency, median (ms) |
| **Tok/s** | Output token throughput per user |
| **Power (W)** | `VDD_CPU_GPU_CV` rail average over the aiperf run window |

**Sweep:** 4 prompt lengths × 3 gen lengths × 20 requests = 240 measurements per model per power mode.

| Prompt tokens | 128 | 512 | 1024 | 2048 |
|---------------|-----|-----|------|------|
| **Gen tokens** | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 |

**Concurrency:** 1 user, 1 request at a time (`--parallel 1`, `--concurrency 1`).



## Prerequisites

- Jetson Orin Nano Super running JetPack R36.x (CUDA 12.x)
- `sudo` access (for `tegrastats`, `nvpmodel`, `jetson_clocks`)
- `tmux` (the script auto-relaunches itself inside a tmux session)

### llama.cpp (required for `--backend llamacpp`)

Build `llama-server` with CUDA for SM\_87 (Jetson Orin Ampere):

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build --config Release -j$(nproc) --target llama-server
```

Binary expected at `~/llama.cpp/build/bin/llama-server`. Override with:
```bash
export LLAMACPP_BIN=/path/to/llama-server
```

### Ollama (required for `--backend ollama`)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama imports the same local GGUF files — no separate model download.

### aiperf (load generator, required)

```bash
python3 -m venv ~/aiperf-env
source ~/aiperf-env/bin/activate
pip install aiperf
```

The script tries `~/venv` then `~/aiperf-env`.

### HuggingFace CLI (for model auto-download)

```bash
pip install huggingface_hub
huggingface-cli login   # required for gated models (Llama-3.2, Gemma3)
```



## Running Benchmarks

The script **auto-launches inside tmux** on every invocation:

```bash
bash bench-non-reasoning.sh [flags]
# → Launched in tmux session 'non-reasoning-bench'
# → Attach with:  tmux attach -t non-reasoning-bench
```

Detach: `Ctrl+B D` · Reattach: `tmux attach -t non-reasoning-bench`

### llama.cpp backend (default)

```bash
bash bench-non-reasoning.sh --power-mode 1   # 25W — recommended sweet spot
bash bench-non-reasoning.sh --power-mode 0   # 15W
bash bench-non-reasoning.sh --power-mode 2   # MAXN
bash bench-non-reasoning.sh --power-mode 3   # 7W
```

### Ollama backend

```bash
bash bench-non-reasoning.sh --backend ollama --power-mode 1
```

### Both backends in one run

```bash
bash bench-non-reasoning.sh --backend both --power-mode 1
```

Runs llama.cpp first, then Ollama. Results land in separate `llamacpp/` and `ollama/` subdirs inside the same artifact dir.

### Quick test — single model

```bash
bash bench-non-reasoning.sh --only smollm2-135m --reqs 2
```

### Resume an interrupted run

```bash
bash bench-non-reasoning.sh --resume artifacts/blog-all-20260602-0139-25w --power-mode 1
```

Already-completed combos (existing `profile_export_aiperf.json`) are skipped automatically.



## Output

```
artifacts/blog-all-YYYYMMDD-HHMM-<mode>/
├── tegrastats.log                    # raw power + thermal (500 ms interval)
├── model_timing.log                  # per-model start/end epochs
├── report.md                         # auto-generated results table
├── llamacpp/                         # llama.cpp results
│   ├── <model>-server.log
│   └── <model>/gen<G>/ctx<P>/
│       ├── profile_export_aiperf.json
│       └── profile_export_aiperf_timeslices.json
└── ollama/                           # Ollama results
    └── <model>/gen<G>/ctx<P>/
        └── profile_export_aiperf.json
```



## Generating Charts

After benchmarks complete, generate the report charts:

```bash
python3 generate_combined_charts.py       # main charts (Figures 1-17)
python3 generate_appendix_ab_charts.py    # Appendix A (4-mode) + B (thermal)
```

Charts are saved to `artifacts/charts/`. The report is `benchmark_report.md`.

> Raw per-cell data is on Hugging Face — see dataset table at the top of `benchmark_report.md`. Local artifact directories can be deleted after charts are generated.

## Arguments

```
bash bench-non-reasoning.sh [OPTIONS]
```

---

#### `--backend <llamacpp|ollama|both>`

- **Optional**
- **Default:** `llamacpp`

Inference backend to use.

| Value | Behaviour |
|-------|-----------|
| `llamacpp` | Launches `llama-server` directly. Binary expected at `~/llama.cpp/build/bin/llama-server`; override with `LLAMACPP_BIN=/path/to/llama-server`. |
| `ollama` | Expects the Ollama daemon to be running. Imports each GGUF from `~/gguf-models/` via a generated Modelfile. |
| `both` | Runs the full llama.cpp sweep first, then Ollama. Results land in separate `llamacpp/` and `ollama/` subdirectories inside the same artifact directory. |

---

#### `--power-mode <N>`

- **Optional**
- **Default:** `0`

Sets the Jetson nvpmodel power envelope before the run starts via `sudo nvpmodel -m N`.

| N | Mode |
|---|------|
| 0 | 15W |
| 1 | 25W |
| 2 | MAXN |
| 3 | 7W |

**Note:** This argument is not saved to disk and is not read back from the artifact directory on `--resume`. If you resume a run without passing `--power-mode`, the hardware will be set to `0` (15W) regardless of what mode the original run used. Always pass the same value you used for the original run.

Switching to or from 7W requires a reboot. The script detects this and exits with instructions rather than benchmarking at the wrong mode.

Shorthand alias: `--maxn` is equivalent to `--power-mode 2`.

---

#### `--reqs <N>`

- **Optional**
- **Default:** `20`

Number of requests per benchmark combo (prompt_len × gen_len cell). Higher values reduce variance; `20` is the default and the value used for published results.

---

#### `--only <model-name>`

- **Optional**
- **Default:** all models

Run a single model instead of the full sweep. The value is matched as a case-insensitive substring against the model name. Valid values: `smollm2-135m`, `smollm2-360m`, `qwen2.5-0.5b`, `qwen3-0.6b`, `llama3.2-1b`, `gemma3-1b`, `gemma3-4b`, `lfm2.5-350m`, `lfm2.5-1.2b`.

Combine with `--resume` to patch a single missing or failed model into an existing run.

---

#### `--resume <dir>`

- **Optional**
- **Default:** none (creates a new timestamped artifact directory)

Reuse an existing artifact directory instead of creating a new one. The script inspects which prompt_len × gen_len combos already have result files (`profile_export_aiperf.json`) for each model and skips them. Only missing combos are benchmarked.

**Required companion argument:** always pass `--power-mode` with the same value used for the original run. The script does not store or recover the original power mode automatically.

```bash
# Resume an interrupted 25W run
bash bench-non-reasoning.sh \
  --resume artifacts/blog-all-YYYYMMDD-HHMM-25w \
  --power-mode 1

# Rerun only one missing model in an existing run
bash bench-non-reasoning.sh \
  --only smollm2-135m \
  --resume artifacts/blog-all-YYYYMMDD-HHMM-25w \
  --power-mode 1
```

---

#### `--skip-smoke`

- **Optional** — flag, no value
- **Default:** off

Skip the smoke test (short generation) that runs before each model's benchmark sweep. The smoke test verifies the server loaded and responds correctly; skipping it saves time but removes the early-failure safety net.

---

#### `--dry-run`

- **Optional** — flag, no value
- **Default:** off

Print the full list of models and combos that would run without executing anything. Useful for verifying `--only` / `--resume` filtering before starting a long run.

---

> **CMA note:** At 7W, Jetson CMA address space fragments after sequential model loads. The script calls `/proc/sys/vm/compact_memory` between models; a reboot may still be needed before running 7W after other modes. nvpmodel itself requires a reboot when switching to or from 7W — the script detects this and exits with instructions rather than benchmarking at the wrong mode.
