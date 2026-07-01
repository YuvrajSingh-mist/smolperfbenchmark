# MoE RPC Cluster Benchmark - Jetson Orin Nano Super (3× 8GB)

Throughput, latency, and energy-efficiency benchmarks for Mixture-of-Experts LLMs too large to fit on a single 8GB Jetson, run across a **3-node llama.cpp RPC cluster**.

One head node pools its memory with two worker Jetsons over the network using llama.cpp's `--rpc` backend, letting each model's weights and KV cache split across three unified-memory boards that individually only have ~7.4GB usable each.

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

| Model | Total / Active Params | Quant | GGUF size |
|-------|------------------------|-------|----------:|
| gpt-oss-20b | 21B / 3.6B | Q4\_K\_M | 11.6 GB |
| Qwen3-30B-A3B | 30B / 3B | Q4\_K\_M | 18.6 GB |
| Granite4.0-H-Small (32B-A9B) | 32B / 9B | Q4\_K\_M | 19.7 GB |

None of these fit on a single 8GB board. Q4\_K\_M was chosen deliberately because bf16/Q8\_0 weight sizes for these three (42–64 GB) exceed even the ~22 GB combined 3-node pool. GGUFs are downloaded **just-in-time** into `~/cluster-gguf-models/` on the head node and deleted after each model's sweep completes — all three combined (~50GB) don't need to fit on disk simultaneously.



## Metrics

| Metric | Description |
|--------|-------------|
| **output tok/J** | OSL / (avg\_power\_W × RL\_p50\_s) — primary efficiency metric |
| **TTFT p50** | Time to first token, median over `--reqs` requests (ms) |
| **ITL p50** | Inter-token latency, median (ms) |
| **Tok/s** | Output token throughput per user |
| **Power (W)** | `VDD_CPU_GPU_CV` rail average over the aiperf run window — **head node only** |
| **rss.log** | VmRSS (kB) of `llama-server` on the head and `rpc-server` on each worker, snapshotted before/after each aiperf run — confirms the model actually split across nodes instead of silently OOM-failing on one board |

**Sweep:** 4 prompt lengths × 3 gen lengths × `--reqs` requests per model per power mode.

| Prompt tokens | 128 | 512 | 1024 | 2048 |
|---------------|-----|-----|------|------|
| **Gen tokens** | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 |

**Concurrency:** 1 user, 1 request at a time (`--parallel 1`, `--concurrency 1`).



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

`RPC_WORKER_HOSTS` and `RPC_WORKER_ADDRS` near the top of `benchmark-moe-rpc.sh` must match your `~/.ssh/config` aliases and each worker's real IP:

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

### llama.cpp with RPC + CUDA (required on all 3 nodes)

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
- Workers: `~/llama.cpp/build/bin/ggml-rpc-server`

Override the head binary path with `LLAMACPP_BIN=/path/to/llama-server`; the worker path is set via `RPC_SERVER_BIN` in the script.

> `Could NOT find OpenSSL` during configure only disables HTTPS in the bundled server — harmless on a private LAN over plain HTTP. `NCCL not found` is also expected and does not block the RPC backend (NCCL only affects same-box multi-GPU tensor parallelism, not cross-node RPC).

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
bash benchmark-moe-rpc.sh [flags]
# → Launched in tmux session 'moe-rpc-bench'
# → Attach with:  tmux attach -t moe-rpc-bench
```

```bash
bash benchmark-moe-rpc.sh --power-mode 1              # 25W — recommended sweet spot
bash benchmark-moe-rpc.sh --power-mode 1 --only gpt-oss
```

**Always pass `--power-mode` explicitly.** Without it, the script defaults to a 4-mode power sweep, and because each cluster GGUF is deleted after its sweep (to stay under disk limits), every mode re-downloads all 3 models from scratch — up to ~200GB of redundant bandwidth. The script prints a warning if you forget.

What happens under the hood per run:
1. SSH into `jetson2`/`jetson3`, kill any stale `ggml-rpc-server`, launch a fresh one, wait for the TCP port to come up.
2. Sync `nvpmodel`/`jetson_clocks` on both workers to match the head's current power mode.
3. For each cluster model: download GGUF just-in-time → launch `llama-server --rpc jetson2:50052,jetson3:50052 -ngl 99` → smoke test → aiperf sweep (with pre/post RSS snapshots across all 3 nodes) → kill server → delete GGUF.
4. Stop `ggml-rpc-server` on both workers.

**Potential perf flags not yet enabled, pending verification:** `-fa on` (Flash Attention) and `-ctk q8_0 -ctv q8_0` (quantized KV cache) exist in this llama.cpp build (`GGML_CUDA_FA=ON` at compile time), and `ggml-rpc-server --cache` (worker-side tensor cache) is a real flag we confirmed via `--help`. None of these are wired into the script yet — they need to be smoke-tested against these specific MoE architectures at 2048 ctx first (unusual attention configs in some MoE models have had FA/quantized-KV compatibility gaps in llama.cpp before), not assumed safe from the CLI help text alone.

### Quick test — single model

```bash
bash benchmark-moe-rpc.sh --only qwen3-30b --reqs 2 --power-mode 1
```

### Resume an interrupted run

```bash
bash benchmark-moe-rpc.sh --resume artifacts/blog-all-YYYYMMDD-HHMM-25w --power-mode 1
```

Already-completed combos (existing `profile_export_aiperf.json`) are skipped automatically.



## Output

```
artifacts/blog-all-YYYYMMDD-HHMM-<mode>/
├── report.md                            # auto-generated results table
└── llamacpp-rpc/
    ├── <model>-<quant>-server.log
    └── <model>-<quant>/gen<G>/ctx<P>/
        ├── profile_export_aiperf.json
        ├── combo_info.json
        ├── tegrastats.log                # head-node power for this combo
        └── rss.log                       # pre/post VmRSS: head + jetson2 + jetson3
```



## Arguments

```
bash benchmark-moe-rpc.sh [OPTIONS]
```

---

#### `--power-mode <N>`

- **Optional, strongly recommended**
- **Default:** none (triggers a 4-mode sweep with a redundant-download warning — see above)

Sets the Jetson nvpmodel power envelope on the head node (and syncs it to both workers) via `sudo nvpmodel -m N`.

| N | Mode |
|---|------|
| 0 | 15W |
| 1 | 25W |
| 2 | MAXN |
| 3 | 7W |

Switching to or from 7W requires a reboot on whichever node isn't already at that mode. Shorthand alias: `--maxn` is equivalent to `--power-mode 2`.

---

#### `--reqs <N>`

- **Optional** — **Default:** `20`

Number of requests per benchmark combo (prompt\_len × gen\_len cell).

---

#### `--only <model-name>`

- **Optional** — **Default:** all 3 models

Case-insensitive substring match. Valid values: `gpt-oss-20b`, `qwen3-30b-a3b`, `granite4-32b-a9b`.

---

#### `--resume <dir>`

- **Optional** — **Default:** none (creates a new timestamped artifact directory)

Reuse an existing artifact directory; skips combos that already have `profile_export_aiperf.json`. Always pass the same `--power-mode` used for the original run.

---

#### `--skip-smoke`

- **Optional** — flag, no value — **Default:** off

Skip the smoke test that runs before each model's benchmark sweep.

---

#### `--dry-run`

- **Optional** — flag, no value — **Default:** off

Print the models/combos that would run without executing anything or touching the RPC workers.

---

#### Cluster topology (script constants, not CLI flags)

Set at the top of `benchmark-moe-rpc.sh`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `RPC_WORKER_HOSTS` | `(jetson2 jetson3)` | SSH aliases (must exist in `~/.ssh/config`, key-auth, no password) |
| `RPC_WORKER_ADDRS` | `("10.10.1.2:50052" "10.10.1.3:50052")` | `ip:port` each worker's `rpc-server` binds to |
| `RPC_SERVER_BIN` | `$HOME/llama.cpp/build/bin/ggml-rpc-server` | Expected path on every worker |
| `RPC_GGUF_DIR` | `$HOME/cluster-gguf-models` | Head-node JIT download scratch dir |

Edit these directly in the script to point at different nodes, ports, or add more workers.

---

> **Disk note:** the 3 cluster models are ~50GB combined at Q4\_K\_M. Since they're downloaded/deleted per-combo rather than pre-fetched in bulk, peak disk usage stays around one model's size (max ~20GB) instead of the full 50GB — check `df -h` on the head node before a full unattended run.
