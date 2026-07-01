# Multi-Node Non-Reasoning + MoE LLM Benchmark - Jetson Orin Nano Super Cluster (3× 8GB)

Extends the [single-node benchmark](../../single-node/non-reasoning-models/) with a **3-node llama.cpp RPC cluster tier** for Mixture-of-Experts models too large to fit on one 8GB Jetson.

One head node (this box) pools its memory with two worker Jetsons over the network using llama.cpp's `--rpc` backend, letting a ~50GB combined GGUF footprint run across three unified-memory boards that individually only have ~7.4GB usable each.

Everything from the single-node script still works unchanged (`llamacpp` / `ollama` / `both` backends, same 8 tiny models, head-node only) — this script adds a fourth backend, `rpc`, for the cluster tier.

Key metric: **output tok/J** = OSL / (avg\_power\_W × RL\_p50\_s)



## Table of Contents

- [Hardware](#hardware)
- [Models](#models)
- [Metrics](#metrics)
- [Cluster Setup](#cluster-setup)
- [Prerequisites](#prerequisites)
- [Running Benchmarks](#running-benchmarks)
- [Output](#output)
- [Arguments](#arguments)



## Hardware

| Component | Detail |
|-----------|--------|
| Nodes | 3× Jetson Orin Nano Super 8GB Developer Kit |
| Roles | 1 head (`jetson`, runs `llama-server` + orchestrates the sweep) + 2 RPC workers (`jetson2`, `jetson3`, run `rpc-server`) |
| Network | Private LAN, static IPs `10.10.1.1` (head), `10.10.1.2` (jetson2), `10.10.1.3` (jetson3) |
| CPU (each) | 6× Arm Cortex-A78AE @ up to 1.728 GHz |
| GPU (each) | NVIDIA Ampere, 1024 CUDA cores, 32 Tensor cores |
| Memory (each) | 8 GB LPDDR5 unified CPU + GPU (~7.4 GB usable) |
| Combined pool | ~22 GB across all 3 nodes for RPC-split weights + KV cache |
| JetPack | R36.4.7 (Ubuntu 22.04, CUDA 12.6) |



## Models

### Single-node tier (`--backend llamacpp|ollama|both`) — head node only

Same 8 models as the [single-node benchmark](../../single-node/non-reasoning-models/README.md#models); see that README for the full table. Included here unchanged for parity/comparison against the cluster runs.

### Cluster tier (`--backend rpc`) — split across all 3 nodes

| Model | Total / Active Params | Quant | GGUF size |
|-------|------------------------|-------|----------:|
| gpt-oss-20b | 21B / 3.6B | Q4\_K\_M | 11.6 GB |
| Qwen3-30B-A3B | 30B / 3B | Q4\_K\_M | 18.6 GB |
| Granite4.0-H-Small (32B-A9B) | 32B / 9B | Q4\_K\_M | 19.7 GB |

None of these fit on a single 8GB board — Q4\_K\_M was chosen deliberately because bf16/Q8\_0 weight sizes for these three (42–64 GB) exceed even the ~22 GB combined 3-node pool. Cluster GGUFs are downloaded **just-in-time** into `~/cluster-gguf-models/` on the head node and deleted after each model's sweep completes — all three combined (~50GB) don't need to fit on disk simultaneously.



## Metrics

Same metrics as the single-node benchmark (output tok/J, TTFT/ITL/tok-s percentiles, power) — see [single-node README](../../single-node/non-reasoning-models/README.md#metrics) for the full table.

**New for the cluster tier:** `rss.log` per combo, capturing VmRSS (in kB) of `llama-server` on the head node and `rpc-server` on each worker, snapshotted immediately before and after each aiperf run. Useful for confirming the model actually split across nodes as expected rather than silently OOM-failing on one board.

**Sweep:** same grid as single-node — 4 prompt lengths × 3 gen lengths × `--reqs` requests per model per power mode.



## Cluster Setup

One-time setup to turn three separate Jetsons into an RPC cluster. Do this before the [Prerequisites](#prerequisites) build steps.

### 1. Passwordless SSH between nodes

On the head node, add both workers to `~/.ssh/config`:

```
Host jetson2
    HostName 10.10.1.2
    User <your-username>
    IdentityFile ~/.ssh/<cluster-key>
    IdentitiesOnly yes

Host jetson3
    HostName 10.10.1.3
    User <your-username>
    IdentityFile ~/.ssh/<cluster-key>
    IdentitiesOnly yes
```

Generate a key and copy it to both workers if you haven't already:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/<cluster-key> -N ""
ssh-copy-id -i ~/.ssh/<cluster-key>.pub <user>@10.10.1.2
ssh-copy-id -i ~/.ssh/<cluster-key>.pub <user>@10.10.1.3
```

Verify both are reachable without a password prompt:

```bash
ssh jetson2 hostname
ssh jetson3 hostname
```

### 2. Match the host aliases in the script

`RPC_WORKER_HOSTS` and `RPC_WORKER_ADDRS` near the top of `benchmark-non-reasoning.sh` must match your `~/.ssh/config` aliases and each worker's real IP:

```bash
RPC_WORKER_HOSTS=(jetson2 jetson3)
RPC_WORKER_ADDRS=("10.10.1.2:50052" "10.10.1.3:50052")
```

Edit these if your node names, IPs, or count of workers differ (the RPC backend isn't limited to exactly 2 workers — add more `host`/`ip:port` pairs to both arrays to scale further).

### 3. Build llama.cpp with RPC + CUDA — on all three nodes

See [Prerequisites](#prerequisites) below. The head needs `llama-server`; each worker only needs `rpc-server`, but it's simplest to build the full target on all three identically.



## Prerequisites

- 3× Jetson Orin Nano Super running JetPack R36.x (CUDA 12.x), networked per [Cluster Setup](#cluster-setup)
- `sudo` access on all 3 nodes (for `tegrastats`, `nvpmodel`, `jetson_clocks`)
- `tmux` on the head node (the script auto-relaunches itself inside a tmux session)

### llama.cpp with RPC + CUDA (required on all 3 nodes for `--backend rpc`)

CUDA toolchain may not be on `PATH` by default on a fresh JetPack image — export it explicitly:

```bash
export PATH="/usr/local/cuda/bin:$PATH"   # run on every node before building

git clone https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Repeat identically on the head node and both workers. Binaries expected at:
- Head: `~/llama.cpp/build/bin/llama-server`
- Workers: `~/llama.cpp/build/bin/rpc-server`

Override the head binary path with `LLAMACPP_BIN=/path/to/llama-server`; the worker path is set via `RPC_SERVER_BIN` in the script.

> `Could NOT find OpenSSL` during configure only disables HTTPS in the bundled server — harmless on a private LAN over plain HTTP. `NCCL not found` is also expected and does not block the RPC backend (NCCL only affects same-box multi-GPU tensor parallelism, not cross-node RPC).

### llama.cpp CPU/GPU only (single-node backend, head node)

If you only want the single-node `llamacpp` backend without the cluster, the same build works — RPC support is additive and doesn't change single-node behavior.

### Ollama (required for `--backend ollama`, head node only)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### aiperf (load generator, required, head node only)

```bash
python3 -m venv ~/aiperf-env
source ~/aiperf-env/bin/activate
pip install aiperf
```

The script tries `~/venv` then `~/aiperf-env`.

### HuggingFace CLI (for model auto-download, head node only)

```bash
pip install --break-system-packages -U "huggingface_hub[cli]"
huggingface-cli login   # only needed for gated models
```



## Running Benchmarks

The script **auto-launches inside tmux** on every invocation, on the head node:

```bash
bash benchmark-non-reasoning.sh [flags]
# → Launched in tmux session 'non-reasoning-bench'
# → Attach with:  tmux attach -t non-reasoning-bench
```

### Single-node tier (unchanged from the original script)

```bash
bash benchmark-non-reasoning.sh --power-mode 1              # llamacpp, 25W
bash benchmark-non-reasoning.sh --backend ollama --power-mode 1
bash benchmark-non-reasoning.sh --backend both --power-mode 1
```

### Cluster tier — RPC backend

```bash
bash benchmark-non-reasoning.sh --backend rpc --power-mode 1
bash benchmark-non-reasoning.sh --backend rpc --power-mode 1 --only gpt-oss
```

**Always pass `--power-mode` explicitly with `--backend rpc`.** Without it, the script defaults to a 4-mode power sweep, and because each cluster GGUF is deleted after its sweep (to stay under disk limits), every mode re-downloads all 3 models from scratch — up to ~150GB of redundant bandwidth. The script prints a warning if you forget.

What happens under the hood per run:
1. SSH into `jetson2`/`jetson3`, kill any stale `rpc-server`, launch a fresh one, wait for the TCP port to come up.
2. Sync `nvpmodel`/`jetson_clocks` on both workers to match the head's current power mode.
3. For each cluster model: download GGUF just-in-time → launch `llama-server --rpc jetson2:50052,jetson3:50052 -ngl 99` → smoke test → aiperf sweep (with pre/post RSS snapshots across all 3 nodes) → kill server → delete GGUF.
4. Stop `rpc-server` on both workers.

### Quick test — single model

```bash
bash benchmark-non-reasoning.sh --backend rpc --only qwen3-30b --reqs 2 --power-mode 1
```

### Resume an interrupted run

```bash
bash benchmark-non-reasoning.sh --backend rpc --resume artifacts/blog-all-YYYYMMDD-HHMM-25w --power-mode 1
```



## Output

```
artifacts/blog-all-YYYYMMDD-HHMM-<mode>/
├── tegrastats.log                       # head-node power + thermal (500 ms interval)
├── report.md                            # auto-generated results table (all backends)
├── llamacpp/                            # single-node llama.cpp results
├── ollama/                              # single-node Ollama results
└── llamacpp-rpc/                        # cluster RPC results
    ├── <model>-<quant>-server.log
    └── <model>-<quant>/gen<G>/ctx<P>/
        ├── profile_export_aiperf.json
        ├── combo_info.json
        ├── tegrastats.log                # head-node power for this combo
        └── rss.log                       # pre/post VmRSS: head + jetson2 + jetson3
```



## Arguments

```
bash benchmark-non-reasoning.sh [OPTIONS]
```

All flags from the single-node script are unchanged — see [single-node README § Arguments](../../single-node/non-reasoning-models/README.md#arguments) for `--reqs`, `--only`, `--resume`, `--skip-smoke`, `--dry-run`, `--power-mode`. Only the additions are documented below.

---

#### `--backend <llamacpp|ollama|both|rpc>`

- **Optional**
- **Default:** `llamacpp`

| Value | Behaviour |
|-------|-----------|
| `llamacpp` | Single-node, head only. Same as the original script. |
| `ollama` | Single-node, head only. Same as the original script. |
| `both` | `llamacpp` then `ollama`, single-node only. |
| `rpc` | 3-node cluster tier. Starts `rpc-server` on `jetson2`/`jetson3` over SSH, runs `llama-server --rpc ...` on the head, benchmarks the 3 cluster MoE models, tears the workers down afterward. |

`rpc` is never bundled into `both` — it's a fundamentally different topology (multi-node vs. single-node) and is opt-in only.

---

#### Cluster topology (script constants, not CLI flags)

Set at the top of `benchmark-non-reasoning.sh`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `RPC_WORKER_HOSTS` | `(jetson2 jetson3)` | SSH aliases (must exist in `~/.ssh/config`, key-auth, no password) |
| `RPC_WORKER_ADDRS` | `("10.10.1.2:50052" "10.10.1.3:50052")` | `ip:port` each worker's `rpc-server` binds to |
| `RPC_SERVER_BIN` | `$HOME/llama.cpp/build/bin/rpc-server` | Expected path on every worker |
| `RPC_GGUF_DIR` | `$HOME/cluster-gguf-models` | Head-node JIT download scratch dir |

Edit these directly in the script to point at different nodes, ports, or add more workers.

---

> **Disk note:** the 3 cluster models are ~50GB combined at Q4\_K\_M. Since they're downloaded/deleted per-combo rather than pre-fetched in bulk, peak disk usage stays around one model's size (max ~20GB) instead of the full 50GB — check `df -h` on the head node before a full unattended run.
