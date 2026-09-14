# smolbenchmark

A collection of reproducible, end-to-end benchmarks for running small open-weight AI models on consumer-grade local hardware (NVIDIA Jetsons, Apple Silicon Macs, Raspberry Pis, phones, tablets, and laptops).

Each subfolder is self-contained with its own benchmark scripts, chart generators, and published reports. No platform assumptions; the goal is to map out what actually runs well on the devices people already own.

## Repository Layout

```
smolbenchmark/
├── README.md                                # this file
├── LICENSE
├── pyproject.toml                           # uv: aiperf 0.11.0 + chart libs
├── uv.lock
├── .python-version                          # 3.12
├── leaderboard/                             # github.io redirect stub → Vercel
├── leaderboard-site/                        # private site clone (gitignored; see its README)
│
├── benchmark-jetson-nano-orin-super/         # NVIDIA Jetson Orin Nano Super 8GB
│   ├── single-node/                         # single-board benchmarks
│   │   ├── non-reasoning-models/            # 8 tiny instruct LLMs (135M-1.2B)
│   │   │   ├── README.md
│   │   │   ├── bench-non-reasoning-models-v2.sh
│   │   │   ├── generate_combined_charts.py
│   │   │   ├── benchmark_report.md
│   │   │   └── artifacts/
│   │   ├── bonsai-models/                   # Bonsai / Ternary-Bonsai family
│   │   │   ├── README.md
│   │   │   ├── benchmark_all_bonsai.sh
│   │   │   ├── generate_combined_charts.py
│   │   │   └── artifacts/
│   │   └── mixture-of-experts/              # planned: single-board MoE tier
│   └── multi-node/                          # 3-board cluster benchmarks
│       └── mixture-of-experts/              # 3-node llama.cpp RPC cluster for MoE models
│           ├── README.md
│           └── benchmark-moe-rpc.sh
│
├── benchmark-mac-mini-m4/                   # Apple Mac Mini M4, 16GB unified memory
│   ├── README.md
│   └── benchmark_non_reasoning.sh
│
├── benchmark-raspberrypi5/                  # Raspberry Pi 5
│   ├── README.md
│   ├── BLOG.md
│   ├── benchmark_all_cpu.sh
│   └── artifacts/
│
└── benchmark-mobile/
    └── Android/                             # OnePlus 10R (Dimensity 8100-Max) via ADB
        ├── README.md
        └── benchmark-non-reasoning.sh
```

## Benchmarks

| Benchmark | Hardware | Models | Report |
|-----------|----------|--------|--------|
| [Non-Reasoning LLM Benchmark](./benchmark-jetson-nano-orin-super/single-node/non-reasoning-models/) | Jetson Orin Nano Super 8GB | 8 tiny instruct LLMs (135M-1.2B) | [smolhub.com](https://www.smolhub.com/posts/jetson-nano-super-benchmark-non-reasoning/) · [local report](./benchmark-jetson-nano-orin-super/single-node/non-reasoning-models/benchmark_report.md) |
| [Bonsai / Ternary-Bonsai Benchmark](./benchmark-jetson-nano-orin-super/single-node/bonsai-models/) | Jetson Orin Nano Super 8GB | Bonsai + Ternary-Bonsai (1.7B / 4B / 8B), Q1_0/Q2_0 | [smolhub.com](https://www.smolhub.com/posts/jetson-orin-nano-super-bonsai-benchmark/) · [local report](./benchmark-jetson-nano-orin-super/single-node/bonsai-models/artifacts/benchmark_report.md) |
| [MoE RPC Cluster Benchmark](./benchmark-jetson-nano-orin-super/multi-node/mixture-of-experts/) | 3× Jetson Orin Nano Super 8GB (RPC cluster) | gpt-oss-20b, Qwen3-30B-A3B, Granite4.0-H-Small (32B-A9B) | results pending |

Single-board Jetson MoE benchmarks (`single-node/mixture-of-experts/`) are planned but not yet implemented.

Live leaderboard: https://smolbenchmark.vercel.app/ (edit/setup: private [`smolbenchmark-leaderboard`](https://github.com/YuvrajSingh-mist/smolbenchmark-leaderboard) README).

## Load generator

Published numbers used **aiperf 0.11.0** (`44addf0`). Do not `pip install aiperf` from PyPI (yanked stub). The pin lives in `pyproject.toml` / `uv.lock`.

## Clone anywhere

Clone this repo to any directory. Python deps are installed with **uv** into `.venv/` at the clone root. Scripts look there first.

```bash
# uv: https://docs.astral.sh/uv/  (macOS: brew install uv)
git clone https://github.com/YuvrajSingh-mist/smolperfbenchmark.git
cd smolperfbenchmark
uv sync                         # aiperf 0.11.0 + hf + chart libs
uv sync --extra mac             # Mac Mini only: also mlx-lm
.venv/bin/aiperf --version      # must print 0.11.0
```

Override with `SMOL_VENV=/path/to/venv` if you keep the environment somewhere else.

## Host this was tested on

Harness setup (clone, venv, aiperf, Android ADB host) was run on:

- MacBook Air M1 (2020)
- Mac Mini M4 (2025), 16 GB unified memory

Jetson and Raspberry Pi scripts run **on the board**. The Mac Mini folder is both a DUT and a host.


## License

MIT

## Fuel the benches

Perf benchmarking burns wall-clock, watts, and a lot of coffee. If these numbers helped you pick a board or a model, fuel the next run:

[![Support me on Ko-fi](https://storage.ko-fi.com/cdn/kofi2.png?v=3)](https://ko-fi.com/O7W120DR8R)
