# Bonsai Jetson Benchmark

Throughput and energy-efficiency benchmarks for the Bonsai and Ternary-Bonsai model families running on a Jetson Orin Nano Super 8GB. Measures tok/s, TTFT, ITL, and tok/J (tokens per joule) across all six model variants at four nvpmodel power envelopes.

---

## Table of Contents

- [Hardware](#hardware)
- [Models](#models)
- [Metrics](#metrics)
- [Installation & Setup](#installation--setup)
- [Running Benchmarks](#running-benchmarks)
  - [Arguments](#arguments)
- [Output](#output)
- [Generating Reports](#generating-reports)

---

## Hardware

| | |
|---|---|
| Board | Jetson Orin Nano Super 8GB |
| Memory | 8 GB LPDDR5 unified (CPU + GPU) |
| Storage | eMMC |
| JetPack | 6.x (CUDA 12.6, SM_87) |

---

## Models

| Model | Quant | Size |
|---|---|---|
| Bonsai-1.7B | Q1_0 | ~237 MB |
| Bonsai-4B | Q1_0 | ~540 MB |
| Bonsai-8B | Q1_0 | ~1.1 GB |
| Ternary-Bonsai-1.7B | Q2_0 | ~300 MB |
| Ternary-Bonsai-4B | Q2_0 | ~700 MB |
| Ternary-Bonsai-8B | Q2_0 | ~1.4 GB |

All models use size-matched Qwen3 tokenizers (1.7B→Qwen3-1.7B, 4B→Qwen3-4B, 8B→Qwen3-8B).

---

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to first token (ms) |
| **ITL** | Inter-token latency (ms) |
| **Tok/s** | Output token throughput per user |
| **Tok/J** | Tokens per joule — tok/s ÷ VDD_CPU_GPU_CV (W). Primary efficiency metric. |

Sweep: 4 prompt lengths × 3 gen lengths × 10 requests = 12 runs per model, 72 per power mode.

| Prompt tokens | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| **Gen tokens** | 128, 256, 512 | 128, 256, 512 | 128, 256, 512 | 128, 256, 512 |

---

## Installation & Setup

### Prerequisites

- Jetson Orin running JetPack 6.x
- `aiperf` installed in `~/venv` (included in Bonsai-demo setup)

### 1. Clone this benchmark repo

```bash
git clone <this-repo>
cd benchmark-jetson-nano-orin-super
```

### 2. Install Bonsai-demo (models + deps)

Clone Bonsai-demo **inside** `benchmark-jetson-nano-orin-super/` so the benchmark script can find it:

```bash
git clone https://github.com/PrismML-Eng/Bonsai-demo.git

# Optional: choose model size to pre-download (8B default)
export BONSAI_MODEL=8B

# Installs deps and downloads models
cd Bonsai-demo && ./setup.sh && cd ..
```

The expected layout after this step:

```
benchmark-jetson-nano-orin-super/
├── Bonsai-demo/
├── bonsai-models/
└── non-reasoning-models/
```

### 3. Build llama-server with CUDA for Jetson

`setup.sh` downloads a CPU-only binary on arm64 — there is no pre-built CUDA release for aarch64. Build from source instead:

```bash
cd Bonsai-demo
bash scripts/build_cuda_linux.sh --archs 87 --output cuda
cd ..
```

`--archs 87` targets SM_87 (Jetson Orin Ampere) specifically. Without it the script compiles a fat binary for desktop GPU architectures (80/86/89/90) that falls back to slow PTX JIT on Orin. The build takes ~20-30 minutes.

The binary will be at `Bonsai-demo/bin/cuda/llama-server`.

### 4. Enter the benchmark directory

```bash
cd bonsai-models
```

---

## Running Benchmarks

All commands assume you are in `benchmark-jetson-nano-orin-super/bonsai-models/`.

```
bash benchmark_all_bonsai.sh [OPTIONS]
```

### Arguments

---

#### `--backend <llamacpp|ollama>`

- **Optional**
- **Default:** `llamacpp`

Inference backend to use. `llamacpp` launches a `llama-server` process directly. `ollama` expects the Ollama daemon to be running and uses its OpenAI-compatible endpoint.

---

#### `--power-mode <N>`

- **Optional**
- **Default:** `2`

Sets the Jetson nvpmodel power envelope before the run starts. The script calls `sudo nvpmodel -m N` at startup.

| N | Mode |
|---|---|
| 0 | 15W |
| 1 | 25W |
| 2 | MAXN_SUPER |
| 3 | 7W |

**Note:** This argument is not saved to disk and is not read back from the artifact directory on `--resume`. If you resume a run without passing `--power-mode`, the hardware will be set to `2` (MAXN_SUPER) regardless of what mode the original run used. Always pass the same value you used for the original run.

---

#### `--reqs <N>`

- **Optional**
- **Default:** `20`

Number of requests per benchmark combo (prompt_len × gen_len cell).

---

#### `--only <model-name>`

- **Optional**
- **Default:** all models

Run a single model instead of the full sweep. The value is matched as a case-insensitive substring against the model name. Valid values: `Bonsai-1.7B`, `Bonsai-4B`, `Bonsai-8B`, `Ternary-Bonsai-1.7B`, `Ternary-Bonsai-4B`, `Ternary-Bonsai-8B`.

Useful for quick tests or to rerun a single model that failed in a previous run (combine with `--resume`).

---

#### `--resume <dir>`

- **Optional**
- **Default:** none (creates a new timestamped artifact directory)

Reuse an existing artifact directory instead of creating a new one. The script inspects which prompt_len × gen_len combos already have result files for each model and skips them. Only missing combos are benchmarked.

Also implies `--skip-download`.

**Required companion argument:** always pass `--power-mode` with the same value used for the original run. The script does not store or recover the original power mode automatically.

```bash
# Resume a 25W run
bash benchmark_all_bonsai.sh \
  --resume artifacts/llamacpp/bonsai-all-YYYYMMDD-HHMM \
  --power-mode 1 \
  --reqs 20

# Rerun only one missing model in an existing run
bash benchmark_all_bonsai.sh \
  --only Ternary-Bonsai-4B \
  --resume artifacts/llamacpp/bonsai-all-YYYYMMDD-HHMM \
  --power-mode 1
```

Always run inside tmux when resuming so the session survives disconnects:

```bash
tmux new-session -d -s bonsai-bench && \
tmux send-keys -t bonsai-bench "cd ~/Desktop/smolbenchmark/benchmark-jetson-nano-orin-super/bonsai-models && \
bash benchmark_all_bonsai.sh --resume artifacts/llamacpp/bonsai-all-YYYYMMDD-HHMM --power-mode 1 --reqs 20" Enter && \
tmux attach -t bonsai-bench
```

Detach: `Ctrl+B D` — reattach: `tmux attach -t bonsai-bench`

---

#### `--skip-download`

- **Optional** — flag, no value
- **Default:** off

Skip the HuggingFace model download check. Use when models are already present at the expected paths and you want to avoid the network round-trip. Implied by `--resume`.

---

#### `--skip-smoke`

- **Optional** — flag, no value
- **Default:** off

Skip the smoke test (short 32/256/512 token generation) that runs before each model's benchmark sweep. The smoke test is a sanity check that the server loaded correctly; skipping it saves ~2 minutes per model but removes the early-failure safety net.

---

#### `--no-lock-clocks`

- **Optional** — flag, no value
- **Default:** off (clocks are locked)

By default the script calls `jetson_clocks` to lock CPU/GPU clocks at their maximum frequency, eliminating DVFS noise from benchmark results. Pass this flag to allow the OS to scale clocks freely (not recommended for reproducible results).

---

#### `--dry-run`

- **Optional** — flag, no value
- **Default:** off

Print the full list of models and combos that would run without executing anything. Useful for verifying `--only` / `--resume` filtering before starting a long run.

---

> **Memory note:** On 8GB Orin, load the 8B model first after a fresh reboot. Physical memory fragments after ~30 min of activity, making the 1 GB contiguous allocation for 8B models fail. The script handles this by ordering 8B first.

---

## Output

Each run creates a timestamped directory under `artifacts/`:

```
artifacts/llamacpp/bonsai-all-YYYYMMDD-HHMM/
├── tegrastats.log               # raw power + thermal samples (500ms interval)
├── model_timing.log             # per-model start/end epochs for power windowing
├── <ModelName>-server.log       # llama-server stdout per model
└── <ModelName>/
    └── gen<G>/
        └── ctx<P>/
            ├── profile_export_aiperf.json
            └── profile_export_aiperf_timeslices.json
```

---

## Generating Reports

### Per-run report

```bash
python3 gen_report.py --artifact llamacpp/bonsai-all-20260527-0200-25W --label "25W"
# writes artifacts/llamacpp/bonsai-all-20260527-0200-25W/report.md
```

### Combined charts (multi-run comparison)

Edit the `RUNS` dict in `gen_combined_charts.py` to include your artifact dirs, then:

```bash
source ~/venv/bin/activate
python3 gen_combined_charts.py
# writes artifacts/charts/*.png
```
