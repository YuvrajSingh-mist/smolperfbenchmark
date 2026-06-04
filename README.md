# smolbenchmark

A collection of reproducible, end-to-end benchmarks for running small open-weight AI models on consumer-grade local hardware (NVIDIA Jetsons, Apple Silicon Macs, Raspberry Pis, phones, tablets, and laptops).

Each subfolder is self-contained with its own benchmark scripts, chart generators, and published reports. No platform assumptions; the goal is to map out what actually runs well on the devices people already own.

## Repository Layout

```
smolbenchmark/
├── README.md                      # this file
├── non-reasoning-models/          # tiny instruct LLMs on Jetson
│   ├── README.MD
│   ├── bench-non-reasoning.sh
│   ├── generate_combined_charts.py
│   └── artifacts/
│   
├── bonsai-models/                 # Bonsai / Ternary-Bonsai family
│   ├── README.md
│   ├── benchmark_all_bonsai.sh
│   ├── gen_report.py
│   └── artifacts/
└── LICENSE
```

## Benchmarks

### [non-reasoning-models](./non-reasoning-models/)

Eight tiny instruct LLMs (135M-1.2B params) benchmarked across four power envelopes on a Jetson Orin Nano Super 8GB. Tests throughput, latency, and energy efficiency (output tok/J) at every prompt x generation combination using both llama.cpp and Ollama backends. Includes a full published report with comparison charts and appendices.

### [bonsai-models](./bonsai-models/)

Bonsai and Ternary-Bonsai model families (1.7B / 4B / 8B) with extreme quantization (Q1_0 / Q2_0) on Jetson hardware. Benchmarks throughput, energy efficiency, and latency across multiple power modes.

More platform folders coming soon (Mac Mini, Raspberry Pi, phones, and tablets).

## Published Reports

| Report | Hardware | Models | Metrics |
|--------|----------|--------|---------|
| [Non-Reasoning LLM Benchmark](https://www.smolhub.com/posts/jetson-nano-super-benchmark-non-reasoning/) | Jetson Orin Nano Super 8GB | 8 tiny instruct LLMs | tok/s, tok/J, TTFT, ITL, power, latency |

## Raw Data

Complete per-cell JSON exports are published on Hugging Face Datasets:

- [jetson-non-reasoning-benchmark-7w](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-7w)
- [jetson-non-reasoning-benchmark-15w](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-15w)
- [jetson-non-reasoning-benchmark-25w](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-25w)
- [jetson-non-reasoning-benchmark-maxn](https://huggingface.co/datasets/YuvrajSingh9886/jetson-non-reasoning-benchmark-maxn)

## Philosophy

Benchmarks should be comparable across devices, reproducible with one script, and honest about measurement caveats. The key metric throughout is **output tok/J** (tokens per joule) because on consumer hardware with limited cooling and battery, energy efficiency is often the real constraint, not peak throughput.

## License

MIT
