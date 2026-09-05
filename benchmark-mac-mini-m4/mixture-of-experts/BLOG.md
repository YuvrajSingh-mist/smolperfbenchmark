---
title: 'Tiny MoE LLM Benchmark: Mac Mini M4 16GB'
date: 2026-07-17
permalink: /posts/mac-mini-m4-moe-benchmark/
author_profile: false
excerpt: "6 small MoE/hybrid LLMs benchmarked on a Mac Mini M4 16GB: llama.cpp Metal vs MLX-LM. SmallThinker-4B-A0.6B hits 130 tok/s at short context; LFM2.5-8B-A1B leads at the 30k-token canonical cell."
header:
  image: /images/blogs/mac-mini-m4-moe-benchmark/preview.png
  teaser: /images/blogs/mac-mini-m4-moe-benchmark/preview.png
  og_image: /images/blogs/mac-mini-m4-moe-benchmark/preview.png
  og_image_width: 1200
  og_image_height: 1855
  og_image_alt: "Mac Mini M4 used for the small MoE LLM inference benchmark"
tags:
  - Apple Silicon
  - Benchmark
  - LLM Inference
  - Edge AI
  - llama.cpp
  - MLX
  - Mixture of Experts
  - Energy Efficiency
  - Mac Mini
---

## Six Small MoE Models × Two Backends: llama.cpp vs MLX-LM on Apple Silicon

**Platform:** Mac Mini M4 16GB  
**CPU:** 10-core Apple M4 · **GPU:** 10-core Apple M4 (Metal) · **Neural Engine:** 16-core (not used by either backend)  
**Memory:** 16 GB unified LPDDR5 shared CPU+GPU+ANE · **OS:** macOS Sequoia  
**Backends:** llama.cpp (Metal, `-ngl 99 --parallel 1 -fa 1 --ignore-eos --no-cache-prompt --cache-ram 0`) · MLX-LM (`mlx_lm.server`, 4-bit, `--prompt-cache-size 0 --decode-concurrency 1`)  
**Runs:** llama.cpp - full sweep, **6 models**. MLX-LM - **3 models** (2 have no published 4-bit MLX conversion or fail to load; see [caveats](#backend-caveats) below)  
**Sweep:** prompt ∈ {256, 512, 1024, 2048, 4096, **30720**} tok × gen ∈ {256, 512, 1024} tok × **20 reqs/combo**  
**Concurrency:** 1 (single-user)  
**Canonical cell:** ctx=30720, gen=1024 - the longest-context, longest-generation point in the sweep  
**Key metric:** **[output tok/J](#appendix-i3)** = [`OSL`](#appendix-i1) ÷ ([`decode_power_W`](#appendix-i5) × [<code>p50_decode_s</code>](#appendix-i1)) - decode-phase energy only

**Raw data:** full per-cell JSON exports (`profile_export_aiperf.json` + `powermetrics.log` + server logs) live in this repo under `artifacts/` - no external dataset was published for this run.

[Table S.1](#table-s1) lists the artifact coverage used throughout this report.

<a id="table-s1"></a>
**Table S.1: Benchmark artifact coverage by backend**

| Backend | Artifact dir | Models | Cells |
|---------|-------------|-------:|------:|
| llama.cpp | [`artifacts/mac-m4-moe-20260704-0115/llamacpp`](artifacts/mac-m4-moe-20260704-0115/llamacpp) | 6 | 108 |
| MLX-LM    | [`artifacts/mac-m4-moe-20260712-1733/mlxlm`](artifacts/mac-m4-moe-20260712-1733/mlxlm) | 3 | 54 |

> `GitHub` repo with all code, scripts, and the plotting notebook that generated every chart below is this repo's [`benchmark-mac-mini-m4/mixture-of-experts`](.) directory.

<a id="backend-caveats"></a>
> **Backend coverage caveat:** `smallthinker-4b-a0.6b` has no published 4-bit MLX conversion, so it only has llama.cpp data. `gemma4-e2b` / `gemma4-e4b` are natively multimodal (text + vision + audio tower weights) - every MLX conversion (unsloth and mlx-community) keeps the full VLM-nested structure that `mlx_lm.server` cannot load (it needs `mlx_vlm.server` instead), so both fail MLX-LM's smoke test. Comparing llama.cpp against a different serving stack for those two would not be a fair backend comparison, so no MLX-LM data exists for them.


## Executive Summary

Six small MoE / hybrid-MoE models were benchmarked on Mac Mini M4 under **llama.cpp (Metal)**, and the three with published 4-bit MLX weights were also benchmarked under **MLX-LM**, for a direct backend comparison. Each model ran *18 combinations* of *prompt × generation length* (20 requests per combo), including a long-context 30,720-token cell.

**Key finding: the fastest model at short context is not the fastest model at long context.** SmallThinker-4B-A0.6B is the clear leader at ctx=256, gen=1024 (**126.42 [tok/s](#appendix-i1)**), but its throughput collapses to **43.5 %** of that by ctx=30720 (55.0 [tok/s](#appendix-i1)) - the steepest drop of any model tested. LFM2.5-8B-A1B, despite starting slower (83.0 [tok/s](#appendix-i1) at ctx=256, gen=1024), retains **76.7 %** of its short-context speed at ctx=30720 and ends up the fastest model at the canonical long-context cell (**63.7 [tok/s](#appendix-i1)**).

**Backend finding: llama.cpp vs MLX-LM is architecture-dependent, not a clean sweep either way.** Granite-4.0-H-Tiny is **~1.3-1.4× faster under MLX-LM** across the sweep; Trinity-Nano-Preview and LFM2.5-8B-A1B are **~5-8 % faster under llama.cpp** on average. There is no single "winning" backend for this model class on Apple Silicon - unlike the CUDA/Jetson case where llama.cpp swept almost every model.

**Sub-5B standouts at ctx=256, llama.cpp:**
- **SmallThinker-4B-A0.6B** - **130.75 [tok/s](#appendix-i1)**, best **short-context** throughput and best [tok/J](#appendix-i3) anywhere in the sweep (**8.56** [tok/J](#appendix-i3) at ctx=256, gen=1024), ~3.6 GB peak [RSS](#appendix-i13) at Q4_K_M
- **Trinity-Nano-Preview** - **90.11 [tok/s](#appendix-i1)** at ctx=256, 2nd-best [tok/J](#appendix-i3) anywhere in the sweep (**6.75** at ctx=512, gen=1024) in the smallest footprint (~4.3 GB peak [RSS](#appendix-i13))

**Canonical long-context cell (ctx=30720, gen=1024), llama.cpp:**
- **LFM2.5-8B-A1B** leads on throughput (**63.73 [tok/s](#appendix-i1)**, 4 % ahead of Trinity-Nano, 40 % ahead of SmallThinker)
- **Trinity-Nano-Preview** edges ahead on [decode tok/J](#appendix-i6) (**1.2403** vs LFM2.5's 1.0251) thanks to lower [decode power](#appendix-i5) (16.09 W vs 16.20 W) at a similar [tok/s](#appendix-i1)

The canonical-cell results are summarized in [Table 1](#table-1) for llama.cpp and [Table 2](#table-2) for MLX-LM.

<a id="table-1"></a>
**Table 1: Throughput and efficiency at the canonical cell (ctx=30720, gen=1024) - llama.cpp, all 6 models**

| Model | Total Params | Active Params | <a href="#appendix-i1" style="color:inherit;text-decoration:none"><code>Output Tok/s</code></a> | <a href="#appendix-i3" style="color:inherit;text-decoration:none"><code>Output Tok/J</code></a> | [Peak RAM](#appendix-i13) (MB) |
|-------|--------:|--------:|-------------:|-------------:|-------------:|
| lfm2.5-8b-a1b          | 8.3B | 1.5B | **63.73** | 1.0251 | 5473 |
| trinity-nano           | 6B   | 1B   | 61.08     | **1.2403** | 4543 |
| smallthinker-4b-a0.6b  | 4B   | 0.6B | 55.04     | 0.8358 | 3716 |
| granite4-h-tiny        | 7B   | 1B   | 45.45     | 0.7829 | 4706 |
| gemma4-e2b             | 5.1B | 2.3B | 42.04     | 0.6760 | 3448 |
| gemma4-e4b             | 8B   | 4.5B | 22.53     | 0.4063 | 5592 |
> Why is `trinity-nano` performing better than `granite4-h-tiny` in terms of [decode tok/J](#appendix-i6)? The reason is that `granite4-h-tiny` has a higher [decode power](#appendix-i5) (16.20 W) than `trinity-nano` (16.09 W), which results in lower energy efficiency despite having a similar [tok/s](#appendix-i1).

<a id="table-2"></a>
**Table 2: Same cell, MLX-LM (3 models with published MLX weights)**

| Model | Total Params | Active Params | <a href="#appendix-i1" style="color:inherit;text-decoration:none"><code>Output Tok/s</code></a> | <a href="#appendix-i3" style="color:inherit;text-decoration:none"><code>Output Tok/J</code></a> | [Peak RAM](#appendix-i13) (MB) |
|-------|--------:|--------:|-------------:|-------------:|-------------:|
| granite4-h-tiny | 7B   | 1B   | **59.41** | **0.9597** | 3683 |
| trinity-nano    | 6B   | 1B   | 55.11     | 1.1992     | 3712 |
| lfm2.5-8b-a1b   | 8.3B | 1.5B | 54.99     | 0.9155     | 4500 |

<!-- † [Output tok/J](#appendix-i3) = [`OSL`](#appendix-i1) ÷ ([`decode_power_W`](#appendix-i5) × [<code>p50_decode_s</code>](#appendix-i1)) - decode-phase energy only, using per-request timestamps from `profile_export.jsonl`. -->

## 1. Test Setup

### 1.1 Hardware

The benchmark host is specified in [Table 3](#table-3).

<a id="table-3"></a>
**Table 3: Hardware configuration**

| Component | Detail |
|-----------|--------|
| Board | Mac Mini M4 (Apple Silicon Desktop) |
| CPU | Apple M4, 10-core |
| GPU | Apple M4, 10-core, Metal 3 |
| Neural Engine | 16-core (unused by either backend under test) |
| Memory | 16 GB unified LPDDR5, shared CPU + GPU + ANE |
| Storage | NVMe SSD |
| Cooling | Active (internal fan); no throttling observed in any run (see [Thermal Summary](#section-25)) |

### 1.2 Software Stack

The exact runtime versions and launch details are listed in [Table 4](#table-4).

<a id="table-4"></a>
**Table 4: Software stack**

| Layer | Version / Detail |
|-------|-----------------|
| Board | Mac Mini M4 16 GB |
| OS | macOS Sequoia |
| llama.cpp | Version 9730 (`e475fa2b5`); Metal backend, built with `-DGGML_METAL=ON`, `-ngl 99 --parallel 1 -c <ctx> -t 1 -fa 1 --prio 2 --mlock --ignore-eos --no-cache-prompt --cache-ram 0` |
| MLX-LM | MLX-LM 0.31.3 with MLX 0.32.0; `mlx_lm.server`, 4-bit MLX weights, `--prompt-cache-size 0 --decode-concurrency 1 --prompt-concurrency 1` |
| Load generator | `aiperf` (NVIDIA AI Performance tool), synthetic prompts at exact target token counts |
| Power telemetry | `powermetrics --samplers cpu_power,gpu_power,thermal -i 50`, **[Combined (CPU+GPU+ANE)](#appendix-i2)** rail |
| RAM telemetry | `ps -o rss=` on the server PID, sampled every 50 ms; report uses the peak |
| Python | pandas, seaborn, matplotlib for chart generation |
| Concurrency | **1 user, 1 request at a time** (`--parallel 1` / `--decode-concurrency 1`) - single-user latency and throughput profile only |
| Context size | 32768 (`max_prompt(30720) + max_gen(1024) + 1024` headroom) |

> **Flag parity note:** llama.cpp's `--ignore-eos` forces exactly `gen` output tokens every request. MLX-LM's server has no equivalent flag - it stops at EOS whenever the model emits one, which is the source of the [OSL](#appendix-i1)-mismatch caveat in [section 4.3](#section-43).

### 1.3 Models Under Test

<a id="table-5"></a>
**Table 5: Models under test**

| Model | Quant | Total Params | Active Params† | Peak [RSS](#appendix-i13) @ ctx=256 (MB) | Architecture‡ |
|-------|-------|----------:|----------:|----------:|-----------|
| SmallThinker-4B-A0.6B  | Q4_K_M | 4B   | 0.6B | 3644 | Standard Transformer MoE (32 experts, top-4) |
| Trinity-Nano-Preview   | Q4_K_M | 6B   | 1B   | 4338 | Standard Transformer MoE (128 experts, top-8 + 1 shared) |
| Granite-4.0-H-Tiny     | Q4_K_M | 7B   | 1B   | 4581 | Hybrid Mamba-2 + Transformer MoE (64 experts, top-6; 36 Mamba / 4 attention layers) |
| LFM2.5-8B-A1B          | Q4_K_M | 8.3B | 1.5B | 5428 | Hybrid Conv + Attention MoE (32 experts, top-4; alternating conv/attention layers) |
| Gemma-4-E2B-it         | Q4_K_M | 5.1B | 2.3B | 3377 | Dense elastic (MatFormer + Per-Layer Embeddings) - **not MoE** |
| Gemma-4-E4B-it         | Q4_K_M | 8B   | 4.5B | 5558 | Dense elastic (MatFormer + Per-Layer Embeddings) - **not MoE** |

> **Disclaimer:** Gemma-4-E2B/E4B are **not MoE** - they're included purely as a dense-model comparison point against the four actual MoE/hybrid-MoE models in this suite.

‡ From each model's `config.json` (expert counts, `layer_types`, and architectures field), not inferred from naming alone.

**Four of six are sparse MoE; the two Gemma-4 models are dense, not MoE.** Granite-4.0-H-Tiny, Trinity-Nano-Preview, LFM2.5-8B-A1B, and SmallThinker-4B-A0.6B all route tokens through a top-k expert subset (see the Architecture column for each one's expert count/top-k). **Gemma-4-E2B/E4B have no expert-routing fields at all** in their config - their `E2B`/`E4B` naming follows Gemma 3n's Effective-Parameter convention for elastic-capacity *dense* models (MatFormer nesting + Per-Layer Embeddings), not mixture-of-experts. Grouping all six under "MoE / hybrid-MoE" earlier in this doc was inaccurate for the Gemma-4 pair specifically - corrected here. 

> **Quantization note:** every model runs **Q4_K_M** (4-bit K-quant medium) under llama.cpp; MLX-LM runs the equivalent published **4-bit** MLX conversion where one exists.

<a id="section-14"></a>
### 1.4 Benchmark Methodology

- For each `model` × `prompt` × `gen` combo, `aiperf` sends 20 single-concurrency requests with synthetic prompts at the exact target token count.
- Power is computed from `powermetrics`' **[Combined (CPU+GPU+ANE)](#appendix-i2)** rail (mW → W) at 50 ms intervals. Per-request nanosecond timestamps from `profile_export.jsonl` (`request_start_ns`, `request_ack_ns`, `request_end_ns`) classify each `powermetrics` sample as **prefill** (start→ack) or **decode** (ack→end). [`decode_power_W`](#appendix-i5) is the median of samples inside decode windows. [`output_tok_J`](#appendix-i3) = [`OSL`](#appendix-i1) ÷ ([`decode_power_W`](#appendix-i5) × [<code>p50_decode_s</code>](#appendix-i1)) - decode-phase energy only, so the high-power prefill spike doesn't inflate the denominator.
- Each server was launched once per model, smoke-tested, then swept through all 18 combos before restart for the next model. `--no-cache-prompt --cache-ram 0` (llama.cpp) / `--prompt-cache-size 0` (MLX-LM) disable KV-cache reuse across requests so prefill cost isn't unfairly cheap on repeated synthetic prompts. This specifically protects the **variance *within* a combo's 20 repeated requests** (`--synthetic-input-tokens-stddev 0`, same target length, different random content, same live server): without it, a later request could get a partial or full prefill discount from an earlier one's cached prefix, understating that request's true [TTFT](#appendix-i11)/prefill cost and skewing the p50/p90/p99 spread reported for the combo. It has nothing to do with variance *between* combos - those already differ by prompt length, so there's no caching concern there.
- **Latency percentile used throughout:** all [`TTFT`](#appendix-i11), [`ITL`](#appendix-i11), and [request latency](#appendix-i11) ([`RL`](#appendix-i11)) values reported in charts, tables, and energy calculations use the **p50 (median)** over the 20 requests per combo, to avoid inflation from occasional slow requests (GC pause, OS scheduling). p90 and p99 are available in `report.md` for tail-latency analysis.
- **[RSS](#appendix-i13) sampling** captures the server process's resident memory every 50 ms during each combo; the report keeps the peak - this is model weights + KV-cache + activation footprint at that specific prompt/gen length, not a fixed "model size."

<a id="section-2"></a>
## 2. Results: Charts

All charts below plot **both backends together**: llama.cpp (all 6 models) and MLX-LM (the 3 shared models — see [backend caveats](#backend-caveats)) are overlaid on the same axes wherever the metric allows it. Every model label in every chart (legend, axis, or heatmap title) is suffixed with its **(total params B=active params B)** from [Table 5](#table-5) - e.g. `granite4-h-tiny (7B=1B)` - so parameter scale is visible without cross-referencing the table.

<a id="section-21"></a>
### 2.1 Throughput vs Prompt Length

[Figure 1](#figure-1) plots [output tok/s](#appendix-i1) by model at *gen=1024* (canonical generation length). Throughput falls monotonically as context grows, but not at the same rate:

<a id="figure-1"></a>
**Figure 1: [Output tok/s](#appendix-i1) vs prompt length - all models, llama.cpp (solid) vs MLX-LM (dashed) (gen=1024)**

![Tok/s vs prompt gen=1024](artifacts/charts/1_tok_s_vs_prompt_gen1024.png)

>`Granite-4.0-H-Tiny` and `Trinity-Nano-Preview` are the two models with 1B active parameters, and yet their [tok/sec](#appendix-i1) differs by an appreciable margin. One is standard transformer MoE, the other is a hybrid Mamba-2 + Transformer MoE design. The reason for the gap could be in the design choice of their kernels.

[Figure 2](#figure-2) plots **prefill** [tok/s](#appendix-i1) (input tokens processed per second during the prompt phase) by model at *gen=1024*, alongside the [output tok/s](#appendix-i1) in [Figure 1](#figure-1). Unlike [output tok/s](#appendix-i1), [prefill tok/s](#appendix-i8) *rises* from ctx=256 to a peak around ctx=1024-4096 before falling back down at ctx=30720 for every model - the reason being the quadratic attention mech in all except `Granite-4.0-H-Tiny` with its Mamba-2 architecture - see [1.4](#section-14):

<a id="figure-2"></a>
**Figure 2: [Prefill tok/s](#appendix-i8) vs prompt length - all models, llama.cpp (solid) vs MLX-LM (dashed) (gen=1024)**

![Prefill tok/s vs prompt gen=1024](artifacts/charts/EI_prefill_tput_vs_prompt_gen1024.png)

[Figure 3](#figure-3) compares [output tok/s](#appendix-i1) and [output tok/J](#appendix-i3) at the canonical cell (ctx=30720, gen=1024) for all 6 models; solid bars represent llama.cpp and hatched bars represent MLX-LM (3 shared models only):

<a id="figure-3"></a>
**Figure 3: Canonical cell: [output tok/s](#appendix-i1) and [tok/J](#appendix-i3) side by side, llama.cpp vs MLX-LM (ctx=30720, gen=1024)**

![Canonical Cell Comparison](artifacts/charts/11_canonical_cell_comparison_ctx30720_gen1024.png)

---

<a id="section-22"></a>
### 2.2 Energy Efficiency

- [Figure 4](#figure-4) plots [output tok/J](#appendix-i3) vs prompt length at *gen=1024*, llama.cpp (solid) vs MLX-LM (dashed). Every model's efficiency decays with context length as [decode power](#appendix-i5) creeps up and per-token throughput drops:

<a id="figure-4"></a>
**Figure 4: [Output tok/J](#appendix-i3) vs prompt length - all models, llama.cpp (solid) vs MLX-LM (dashed) (gen=1024)**

![Output Tok/J vs prompt](artifacts/charts/2_tok_j_vs_prompt_gen1024.png)

- [Figure 4b](#figure-4b) shows the best [output tok/J](#appendix-i3) per model, searched across all 18 combos and both backends (solid bar = llama.cpp, hatched bar = MLX-LM). Each bar is labeled with the winning `ctx/gen` cell (e.g. `@ 256/1024` = ctx=256, gen=1024) so the source combo isn't hidden. Short-context, long-generation cells win almost everywhere (see [section 3.1](#section-31)):

<a id="figure-4b"></a>
**Figure 4b: Best [output tok/J](#appendix-i3) per model across all 18 combinations, llama.cpp (solid) vs MLX-LM (hatched)**

![Best Output Tok/J Bar](artifacts/charts/3_best_tok_j_bar.png)

- [Table 6](#table-6) reports the llama.cpp ÷ MLX-LM speed and efficiency ratios at the canonical cell and across the full sweep. Values > 1× mean llama.cpp is faster or more efficient; values < 1× mean MLX-LM is faster or more efficient. The canonical cell is ctx=30720, gen=1024; the full-sweep range is the min/max ratio across all 18 combos for each model.

<a id="table-6"></a>
**Table 6: llama.cpp vs MLX-LM - <a href="#appendix-i1" style="color:inherit;text-decoration:none"><code>tok/s</code></a> and <a href="#appendix-i3" style="color:inherit;text-decoration:none"><code>tok/J</code></a> ratios (llama.cpp ÷ MLX-LM), canonical cell + full-sweep range**

| Model | Total Params | Active Params | [Tok/s](#appendix-i1) ratio (canonical) | [Tok/s](#appendix-i1) ratio (full-sweep range) | [Tok/J](#appendix-i3) ratio (canonical) | [Tok/J](#appendix-i3) ratio (full-sweep range) |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| Granite-4.0-H-Tiny | 7B   | 1B   | 0.77× | 0.72×-0.77× | 0.82× | 0.50×-0.85× |
| LFM2.5-8B-A1B      | 8.3B | 1.5B | **1.16×** | 1.01×-1.17× | **1.12×** | 0.75×-1.12× |
| Trinity-Nano-Preview | 6B | 1B | **1.11×** | 1.06×-1.11× | 1.03× | 0.83×-1.03× |

> Granite-4.0-H-Tiny is consistently **faster under MLX-LM** on throughput - **1.3-1.4× the other way** on [tok/s](#appendix-i1) a; LFM2.5 and Trinity-Nano are consistently **faster under llama.cpp** on both [tok/s](#appendix-i1) and [tok/J](#appendix-i3), by a smaller margin. Full per-cell ratio charts are in [**Appendix C**](#appendix-c).

- [Figure 5](#figure-5) plots [prefill tok/J](#appendix-i6) (input tokens per joule of [prefill energy](#appendix-i5)) vs prompt length at *gen=1024*, llama.cpp (solid) vs MLX-LM (dashed):

<a id="figure-5"></a>
**Figure 5: [Prefill tok/J](#appendix-i6) vs prompt length - all models, llama.cpp (solid) vs MLX-LM (dashed) (gen=1024)**

![Prefill tok/J vs prompt gen=1024](artifacts/charts/22e_prefill_tokj_vs_prompt_gen1024.png)

- [Figure 6](#figure-6) plots [total tok/J](#appendix-i6) ((input + output) tokens per joule of [total request energy](#appendix-i4)) vs prompt length at *gen=1024*, llama.cpp (solid) vs MLX-LM (dashed). It grows with context because the prompt dominates the ([ISL](#appendix-i1)+[OSL](#appendix-i1)) numerator while total energy grows more slowly:

<a id="figure-6"></a>
**Figure 6: [Total tok/J](#appendix-i6) vs prompt length - all models, llama.cpp (solid) vs MLX-LM (dashed) (gen=1024)**

![Total tok/J vs prompt gen=1024](artifacts/charts/22g_total_tokj_vs_prompt_gen1024.png)

> Full [tok/s](#appendix-i1) and [tok/J](#appendix-i3) charts for gen=256 and gen=512 are also in `artifacts/charts/` (`1_tok_s_vs_prompt_gen{256,512}.png`, `2_tok_j_vs_prompt_gen{256,512}.png`).

---

<a id="section-23"></a>
### 2.3 Latency

[Figure 8](#figure-8) shows [`TTFT`](#appendix-i11) p50 at the canonical cell, where solid bars represent llama.cpp and hatched bars represent MLX-LM. [TTFT](#appendix-i11) scales with prompt length and is the dominant cost at ctx=30720 (34-113 s depending on model):

<a id="figure-8"></a>
**Figure 8: [`TTFT`](#appendix-i11) p50 by model, llama.cpp vs MLX-LM (ctx=30720, gen=1024)**

![TTFT by model](artifacts/charts/5_ttft_vs_prompt.png)

[Figure 9](#figure-9) compares [`ITL`](#appendix-i11) *(inter-token latency)* p50 at the canonical cell for llama.cpp vs MLX-LM. Lower is better; this is the per-token decode cost once prefill is excluded:

<a id="figure-9"></a>
**Figure 9: [`ITL`](#appendix-i11) p50 by model, llama.cpp vs MLX-LM (ctx=30720, gen=1024)**

![ITL Comparison](artifacts/charts/8_itl_compare.png)

[Figure 10](#figure-10) compares [request latency](#appendix-i11) (E2E) p50 at the canonical cell for llama.cpp vs MLX-LM. This is the total time from request start to last token received and is dominated by [TTFT](#appendix-i11) at this context length:

<a id="figure-10"></a>
**Figure 10: [Request latency](#appendix-i11) (E2E) p50 by model, llama.cpp vs MLX-LM (ctx=30720, gen=1024)**

![Request Latency Comparison](artifacts/charts/10_request_latency_compare.png)

---

<a id="section-24"></a>
### 2.4 [Prefill Throughput](#appendix-i8)

[Figure 11](#figure-11) compares [prefill tok/s](#appendix-i8) averaged across all prompt lengths at *gen=1024* for llama.cpp vs MLX-LM:

<a id="figure-11"></a>
**Figure 11: [Prefill throughput](#appendix-i8) by model, llama.cpp vs MLX-LM (gen=1024, avg over all prompt lengths)**

![Prefill Comparison](artifacts/charts/9_prefill_compare.png)

---

<a id="section-25"></a>
### 2.5 Power Draw and Thermal Summary

[Figure 12](#figure-12) compares average total [Combined (CPU+GPU+ANE)](#appendix-i2) power per model for llama.cpp vs MLX-LM; [Tables 7](#table-7) and [8](#table-8) provide the backend-specific values and thermal status:

<a id="figure-12"></a>
**Figure 12: Average Combined power draw per model, llama.cpp vs MLX-LM, averaged across the full sweep**

![Avg Power Bar](artifacts/charts/4_avg_power_bar.png)

<a id="table-7"></a>
**Table 7: Thermal summary - llama.cpp**

| Model | Total Params | Active Params | Avg Total W | Throttled |
|-------|--------:|--------:|---:|:---:|
| gemma4-e2b            | 5.1B | 2.3B | 14.76 | No |
| gemma4-e4b             | 8B   | 4.5B | 15.09 | No |
| granite4-h-tiny        | 7B   | 1B   | 14.37 | No |
| lfm2.5-8b-a1b          | 8.3B | 1.5B | 14.93 | No |
| smallthinker-4b-a0.6b  | 4B   | 0.6B | 14.71 | No |
| trinity-nano           | 6B   | 1B   | 13.44 | No |

<a id="table-8"></a>
**Table 8: Thermal summary - MLX-LM**

| Model | Total Params | Active Params | Avg Total W | Throttled |
|-------|--------:|--------:|---:|:---:|
| granite4-h-tiny | 7B   | 1B   | 12.25 | No |
| lfm2.5-8b-a1b   | 8.3B | 1.5B | 13.97 | No |
| trinity-nano    | 6B   | 1B   | 12.49 | No |

> macOS `powermetrics`' `thermal` sampler on the M4 does not expose a per-core junction temperature the way `tegrastats` does on Jetson, so no CPU/GPU °C columns are reported here - only the throttling flag, derived from whether decode throughput degraded mid-run. **No model throttled at any point in either sweep.**

- Every model in this suite draws **12-15 W** average Combined power - the M4's unified power envelope is far flatter across model sizes than the Jetson's CUDA rail was across power modes, because there is no separate governor to widen the spread.
- MLX-LM draws **noticeably less average power** than llama.cpp for the same three models (12.25-13.97 W vs 13.44-14.93 W) - consistent with MLX being a purpose-built Apple Silicon framework with less server-side overhead than a cross-platform Metal backend.

<a id="section-26"></a>
### 2.6 GPU Usage

`powermetrics`' own GPU-usage section (`GPU HW active frequency`, `GPU idle residency`) is a signal none of the [tok/s](#appendix-i1), [tok/J](#appendix-i3), or power charts above use directly - it says how hard the Metal GPU itself is working, independent of how many output tokens that work produces. Charted here for the **canonical combo only** (ctx=30720, gen=1024), both backends - llama.cpp all 6 models (solid), MLX-LM the 3 shared models (dashed) - a 30k-token prompt means most of each request's wall time is prefill, which is exactly where GPU load is easiest to see.

[Figure 13](#figure-13) shows the frequency trace over the run; [Figure 14](#figure-14), [Table 9](#table-9), and [Table 9b](#table-9b) summarize active frequency and residency by backend.

<a id="figure-13"></a>
**Figure 13: GPU active frequency over the canonical-combo run, llama.cpp (solid) vs MLX-LM (dashed) (ctx=30720, gen=1024)**

![GPU active frequency over run](artifacts/charts/gpu_freq_over_run_canonical.png)

Each of the 20 profiled requests in [Figure 13](#figure-13) shows up as one ramp: the sharp climb to ~1570 MHz is prefill (the GPU chews through the 30,720-token prompt at its highest clock), and the plateau/dip that follows is decode (lower, steadier load per output token). The deep dip at the very start (and matching one at the very end) is the server sitting idle before the first request / after the last one - not part of any request's actual compute. SmallThinker-4B-A0.6B and Trinity-Nano-Preview dip lowest during decode (down to ~1420-1440 MHz) - consistent with them being the two fastest, most memory-bandwidth-bound decoders in the suite ([section 3.4](#section-34)), which need less sustained GPU clock per output token than the heavier models. The three MLX-LM dashed lines track their llama.cpp solid twins almost exactly in shape (same ramp-then-plateau pattern per request) - the GPU clock governor's behaviour is a hardware/OS property, not a backend one; what differs between backends is how *much* GPU-second each request costs, not how the clock schedules within a request.

<a id="figure-14"></a>
**Figure 14: GPU average active frequency and active residency, llama.cpp vs MLX-LM, canonical combo**

![GPU utilization bars](artifacts/charts/gpu_utilization_bars_canonical.png)

<a id="table-9"></a>
**Table 9: GPU utilization at the canonical cell, llama.cpp**

| Model | Total Params | Active Params | Avg GPU freq while active (MHz) | GPU active residency (%) |
|-------|--------:|--------:|---:|---:|
| gemma4-e4b            | 8B   | 4.5B | 1500 | 99.74 |
| gemma4-e2b            | 5.1B | 2.3B | 1494 | 99.58 |
| granite4-h-tiny       | 7B   | 1B   | 1497 | 99.57 |
| lfm2.5-8b-a1b         | 8.3B | 1.5B | 1475 | 99.48 |
| smallthinker-4b-a0.6b | 4B   | 0.6B | 1453 | 99.45 |
| trinity-nano          | 6B   | 1B   | 1490 | 99.19 |

<a id="table-9b"></a>
**Table 9b: GPU utilization at the canonical cell, MLX-LM (3 shared models)**

| Model | Total Params | Active Params | Avg GPU freq while active (MHz) | GPU active residency (%) |
|-------|--------:|--------:|---:|---:|
| trinity-nano          | 6B   | 1B   | 1503 | 96.88 |
| lfm2.5-8b-a1b         | 8.3B | 1.5B | 1493 | 99.19 |
| granite4-h-tiny       | 7B   | 1B   | 1492 | 97.82 |

**The GPU is busy nearly the entire canonical-cell run for every model under both backends (96.9-99.7 % active residency)** - at ctx=30720 there simply isn't enough idle time between prefill and decode for the GPU to go idle mid-request. Average active clock also barely varies across models or backends (1453-1503 MHz, a 3.4 % spread) despite their 2.7× spread in [output tok/s](#appendix-i1) ([Table 1](#table-1)) - throughput differences at this cell come from how much *work per token* each architecture needs, not from the GPU running at a different clock for different models or backends. MLX-LM's active residency is 0.3-2.3 points lower than llama.cpp's for the same 3 models (granite4-h-tiny: 97.82 % vs 99.57 %, -1.75 pt; lfm2.5-8b-a1b: 99.19 % vs 99.48 %, -0.29 pt; trinity-nano: 96.88 % vs 99.19 %, -2.31 pt), consistent with MLX-LM's lower average power in [Table 8](#table-8) - marginally more idle GPU time between samples. The M4's peak GPU clock observed anywhere in this dataset is 1578 MHz; every model's canonical-cell average sits within 8 % of that ceiling, whichever backend is serving it.

<a id="section-27"></a>
### 2.7 Processor Usage and Frequency

The same canonical combo's `powermetrics` log also reports per-cluster CPU frequency and residency - `E-Cluster` (6 efficiency cores) and `P-Cluster` (4 performance cores) on the M4's 10-core CPU. This tells a very different story from the GPU: it shows where request scheduling / KV-cache bookkeeping actually runs, separate from the Metal-GPU matmuls charted above. Charted for both backends - llama.cpp all 6 models, MLX-LM the 3 shared models.

[Figure 15](#figure-15), [Table 10](#table-10), and [Table 10b](#table-10b) compare cluster residency; [Figures 15](#figure-16) and [16](#figure-17) show P-Cluster frequency.

<a id="figure-15"></a>
**Figure 15: E-Cluster vs P-Cluster active residency, llama.cpp vs MLX-LM, canonical combo**

![Cluster active residency](artifacts/charts/cluster_active_residency_canonical.png)

<a id="table-10"></a>
**Table 10: CPU cluster active residency at the canonical cell, llama.cpp**

| Model | Total Params | Active Params | E-Cluster active (%) | P-Cluster active (%) |
|-------|--------:|--------:|---:|---:|
| gemma4-e4b            | 8B   | 4.5B | 1.01 | 95.43 |
| granite4-h-tiny       | 7B   | 1B   | 1.22 | 95.26 |
| gemma4-e2b            | 5.1B | 2.3B | 1.10 | 95.32 |
| lfm2.5-8b-a1b         | 8.3B | 1.5B | 1.25 | 95.25 |
| smallthinker-4b-a0.6b | 4B   | 0.6B | 1.31 | 95.24 |
| trinity-nano          | 6B   | 1B   | 1.36 | 94.91 |

<a id="table-10b"></a>
**Table 10b: CPU cluster active residency at the canonical cell, MLX-LM (3 shared models)**

| Model | Total Params | Active Params | E-Cluster active (%) | P-Cluster active (%) |
|-------|--------:|--------:|---:|---:|
| trinity-nano          | 6B   | 1B   | 1.25 | 96.12 |
| granite4-h-tiny       | 7B   | 1B   | 1.67 | 95.60 |
| lfm2.5-8b-a1b         | 8.3B | 1.5B | 1.33 | 95.34 |

**Inference runs almost entirely on the 4 performance cores under both backends.** The 6 efficiency cores sit at ~1.2-1.7 % active residency for every model/backend pair - macOS's scheduler keeps llama.cpp's single-threaded serving loop (`-t 1`) and MLX-LM's request handling on the P-cluster, leaving the E-cluster for background OS work only. This mirrors the GPU picture in [2.6](#section-26): whichever resource is doing the actual token-generation work is saturated (P-cluster ~95-96 %, GPU ~97-99.7 %), while everything else on the chip is essentially untouched, for llama.cpp and MLX-LM alike.

<a id="figure-16"></a>
**Figure 16: Average P-Cluster (performance-core) frequency while active, llama.cpp vs MLX-LM, canonical combo**

![P-Cluster frequency bar](artifacts/charts/p_cluster_freq_canonical.png)

<a id="figure-17"></a>
**Figure 17: P-Cluster active frequency over the canonical-combo run, llama.cpp (solid) vs MLX-LM (dashed)**

![P-Cluster frequency over run](artifacts/charts/p_cluster_freq_over_run_canonical.png)

**Every model pins the P-cluster at ~3.99-4.03 GHz for essentially the entire run, under both backends** ([Figure 17](#figure-17) - all nine model/backend lines sit almost directly on top of each other: llama.cpp averages 4021-4030 MHz across the 3 shared models, MLX-LM averages 3990-4018 MHz on the same 3), with no per-request ramp like the GPU frequency in [Figure 13](#figure-13). This is the clearest evidence in this benchmark that **Apple Silicon has no equivalent to Jetson's `nvpmodel` governor**: the P-cluster does not step down clock between requests, during prefill vs. decode, between a 135 MB-class and multi-GB-class model, or between backends - it simply runs at (very close to) its maximum frequency for the whole combo, and the OS's own DVFS is the only thing standing between this and a flat line at exactly one value.

<a id="section-28"></a>
### 2.8 [RSS](#appendix-i13) (Memory) Usage

The third per-process signal sampled alongside `powermetrics` is **[RSS](#appendix-i13)** (resident set size) - `ps -o rss=` on the server PID, taken every 50 ms for the whole combo ([Appendix I.13](#appendix-i13)). It's model weights + KV-cache + activations together, not a fixed "model size" number - and at the canonical combo's 30,720-token prompt, the KV-cache component is as large as it gets anywhere in this sweep. Both backends are charted: llama.cpp all 6 models (solid), MLX-LM the 3 shared models (dashed).

[Figure 18](#figure-18) traces [RSS](#appendix-i13) over the run, while [Figure 19](#figure-19) compares peak [RSS](#appendix-i13) by model and backend.

<a id="figure-18"></a>
**Figure 18: Server process [RSS](#appendix-i13) over the canonical-combo run, llama.cpp (solid) vs MLX-LM (dashed) (ctx=30720, gen=1024)**

![RSS over run](artifacts/charts/rss_over_run_canonical.png)

<a id="figure-19"></a>
**Figure 19: Peak [RSS](#appendix-i13) at the canonical combo, llama.cpp vs MLX-LM**

![Peak RSS bar](artifacts/charts/rss_peak_canonical.png)

Peak values in [Figure 19](#figure-19) match the **[Peak RAM](#appendix-i13)** columns of [Table 1](#table-1) (llama.cpp) and [Table 2](#table-2) (MLX-LM) exactly (3.4-5.6 GB llama.cpp, 3.7-4.5 GB MLX-LM). **MLX-LM's peak [RSS](#appendix-i13) is lower than llama.cpp's for all 3 shared models** (granite4-h-tiny: 3683 MB vs 4706 MB; lfm2.5-8b-a1b: 4500 MB vs 5473 MB; trinity-nano: 3712 MB vs 4543 MB) - a 17.8-21.7 % reduction, plausibly from MLX's array framework packing the KV-cache more tightly than GGML's buffer allocator at this context length.

What [Figure 18](#figure-18) adds is the *shape*: under **llama.cpp**, Granite-4.0-H-Tiny and Trinity-Nano-Preview show a visible sawtooth ([RSS](#appendix-i13) drops ~100-180 MB between requests, then climbs back), while LFM2.5-8B-A1B, SmallThinker-4B-A0.6B, and both Gemma-4 variants hold a completely flat line for the entire run. With `--no-cache-prompt --cache-ram 0` disabling KV-cache reuse across requests for every model alike, the sawtooth implies Granite's and Trinity's llama.cpp allocators actually release the freed KV-cache buffer back between requests, while the others keep the high-water-mark buffer allocated (cheaper to reuse than to free and reallocate every request, at the cost of a flat peak instead of an average that's lower than peak). **Under MLX-LM the pattern partially inverts**: granite4-h-tiny's dashed line is essentially flat (no sawtooth), while trinity-nano's dashed line still shows a small periodic dip - the allocator behaviour that drives this sawtooth is backend- and model-specific, not a property of the model architecture alone.

**All six models stay well inside the Mac Mini M4's 16 GB unified pool even at this longest-context cell, under either backend** - the largest footprint (Gemma-4-E4B under llama.cpp, 5.6 GB) leaves more than 10 GB of headroom for the OS and other processes.

## 3. Analysis

<a id="section-31"></a>
### 3.1 Higher [tok/sec](#appendix-i1) != efficient model ([tok/J](#appendix-i3))

[Tok/s](#appendix-i1) and [tok/J](#appendix-i3) side by side - see [Figure 3](#figure-3) and [Figure 4b](#figure-4b). The fastest cell for a model is not always its most efficient one.

- **SmallThinker-4B-A0.6B is the sharpest example.** At ctx=256 it is fastest at gen=256 (130.75 [tok/s](#appendix-i1)) but *least* efficient of its own three gen lengths there (7.38 [tok/J](#appendix-i3)); at gen=1024 (same ctx=256) it is slightly slower (126.42 [tok/s](#appendix-i1)) but **most** efficient (8.56 [tok/J](#appendix-i3)) - a longer generation amortizes the fixed [prefill energy](#appendix-i5) over more output tokens.

<a id="table-11"></a>
**Table 11: SmallThinker-4B-A0.6B - [tok/s](#appendix-i1) vs [tok/J](#appendix-i3) crossover at ctx=256 vs ctx=30720**

| Gen length | [Tok/s](#appendix-i1) @ ctx=256 | [Tok/J](#appendix-i3) @ ctx=256 | [Tok/s](#appendix-i1) @ ctx=30720 | [Tok/J](#appendix-i3) @ ctx=30720 |
|-----------:|---------:|---------:|---------:|---------:|
| 256  | 130.75 | 7.38 | 55.47 | 0.25 |
| 512  | 129.51 | 7.73 | 55.34 | 0.47 |
| 1024 | 126.42 | **8.56** | 55.04 | 0.84 |

- Unlike the Jetson power-mode case, there is no clock/power tradeoff driving this on Apple Silicon (single operating point) - the driver here is purely **generation length amortizing [prefill energy](#appendix-i5)**, and separately, **context length collapsing throughput** (see [3.2](#section-32)).

<a id="section-32"></a>
### 3.2 [Context-Length Retention](#appendix-i9): the short-context leader is not the long-context leader

<a id="figure-7"></a>
**Figure 7: [Output tok/s](#appendix-i1) retained at ctx=30720 vs ctx=256 (gen=1024)**

![Context length retention](artifacts/charts/context_length_retention_bar.png)

<a id="table-12"></a>
**Table 12: [Context-length retention](#appendix-i9), llama.cpp (gen=1024)**

| Model | Total Params | Active Params | [Tok/s](#appendix-i1) @ ctx=256 | [Tok/s](#appendix-i1) @ ctx=30720 | Retained |
|-------|--------:|--------:|---------:|---------:|---------:|
| granite4-h-tiny        | 7B   | 1B   | 50.6  | 45.5 | **89.9 %** |
| lfm2.5-8b-a1b          | 8.3B | 1.5B | 83.0  | 63.7 | 76.7 % |
| trinity-nano           | 6B   | 1B   | 86.8  | 61.1 | 70.4 % |
| gemma4-e4b             | 8B   | 4.5B | 30.4  | 22.5 | 74.0 % |
| gemma4-e2b             | 5.1B | 2.3B | 58.1  | 42.0 | 72.4 % |
| smallthinker-4b-a0.6b  | 4B   | 0.6B | 126.4 | 55.0 | **43.5 %** |

**SmallThinker-4B-A0.6B wins every short-context cell but loses the canonical long-context cell to LFM2.5-8B-A1B and Trinity-Nano-Preview** - it retains less than half its ctx=256 throughput by ctx=30720, the steepest drop of any model tested, while Granite-4.0-H-Tiny (the slowest model at short context) is the *most* context-stable, retaining 90 %. This is the single most important nuance in this benchmark: **"fastest model" is not a fixed label - it depends on the context length of the workload.**

The [retention crossover](#appendix-i9) is visualized in [Figure 7](#figure-7) and tabulated in [Table 12](#table-12).

### 3.3 [Best Total tok/J](#appendix-i10) per Model - Ranked by Peak Throughput (llama.cpp)

[Table 13](#table-13) reports the llama.cpp maxima, and [Table 14](#table-14) reports the MLX-LM maxima.

<a id="table-13"></a>
**Table 13: [Best total tok/J](#appendix-i10) per model, searched across all 18 llama.cpp combos**

| Model | Total Params | Active Params | [Best total tok/J](#appendix-i10) | At ctx / gen |
|-------|--------:|--------:|-----------------:|---------------|
| smallthinker-4b-a0.6b  | 4B   | 0.6B | **32.74** | 4096 / 256 |
| trinity-nano           | 6B   | 1B   | **31.29** | 4096 / 256 |
| lfm2.5-8b-a1b          | 8.3B | 1.5B | 19.94     | 4096 / 256 |
| gemma4-e2b             | 5.1B | 2.3B | 15.10     | 4096 / 256 |
| granite4-h-tiny        | 7B   | 1B   | 15.32     | 30720 / 256 |
| gemma4-e4b             | 8B   | 4.5B | 7.62      | 30720 / 256 |

> [Total tok/J](#appendix-i6) = ([`ISL`](#appendix-i1) + [`OSL`](#appendix-i1)) / (avg\_power\_W × [`RL`](#appendix-i11)\_p50\_s) - see [Appendix I.6](#appendix-i6). Peaks at long prompt / short generation for most models because the prompt dominates the numerator while decode stays cheap; Granite and Gemma-4-E4B peak at the very longest context instead, reflecting their flatter power curves.

<a id="table-14"></a>
**Table 14: [Best total tok/J](#appendix-i10) per model, MLX-LM (3 models)**

| Model | Total Params | Active Params | [Best total tok/J](#appendix-i10) | At ctx / gen |
|-------|--------:|--------:|-----------------:|---------------|
| trinity-nano    | 6B   | 1B   | **31.47** | 4096 / 256 |
| lfm2.5-8b-a1b   | 8.3B | 1.5B | 23.72     | 4096 / 256 |
| granite4-h-tiny | 7B   | 1B   | 21.02     | 2048 / 512 |

<a id="section-34"></a>
### 3.4 Latency Characteristics

**[`TTFT`](#appendix-i11) scales near-linearly with prompt length.** At ctx=256 every model prefills in well under a second; at the canonical ctx=30720 that grows to **34.6-112.6 s** depending on model - by far the dominant term in end-to-end [request latency](#appendix-i11) at this context length (see [Figure 10](#figure-10)).

**Inter-token latency ([`ITL`](#appendix-i11)) p50** is the median per-token decode cost. At the canonical cell:

- Gemma-4-E4B has by far the highest [ITL](#appendix-i11) (44.4 ms/tok, i.e. ~22.5 [tok/s](#appendix-i1) decode) - consistent with it being the largest model and the slowest at every cell tested.
- SmallThinker-4B-A0.6B, Trinity-Nano-Preview, and LFM2.5-8B-A1B cluster tightly (15.7-18.2 ms/tok) at the canonical cell despite having very different [ITL](#appendix-i11) at short context - context length compresses the gap between the faster models more than it widens it.

**[Peak RAM](#appendix-i13) ([RSS](#appendix-i13))** scales with context length as the KV-cache grows, on top of a per-model baseline footprint. At ctx=256, footprints range from **3.4 GB** (Gemma-4-E2B) to **5.6 GB** (Gemma-4-E4B) - all comfortably inside the Mac Mini M4's 16 GB unified pool even before accounting for the OS and other processes. Full [peak-RAM](#appendix-i13) heatmaps are in [Appendix I.2](#appendix-i2-charts) (`i2g_peak_ram.png`).

### 3.5 [Prefill vs Decode Energy](#appendix-i5) Split

[Figure 20](#figure-20) shows how [request energy](#appendix-i4) divides between prefill and decode, while [Figure 21](#figure-21) normalizes [decode energy per output token](#appendix-i7) at the canonical cell.

<a id="figure-20"></a>
**Figure 20: [Request energy](#appendix-i4) split - prefill % vs decode % (median across all combos)**

![Prefill/decode energy split](artifacts/charts/E_prefill_decode_energy_split.png)

<a id="figure-21"></a>
**Figure 21: [Decode energy per output token](#appendix-i7) (mJ), canonical cell**

![mJ per output token by model](artifacts/charts/E_mj_per_output_token.png)

- At short-to-medium context, [decode energy](#appendix-i5) dominates the request (many output tokens, one prefill). As context grows toward the 30,720-token canonical cell, [prefill energy](#appendix-i5) takes over the request-energy budget for every model - directly visible in [Figure 6](#figure-6) ([total tok/J](#appendix-i6) rising with context while [decode tok/J](#appendix-i6) in [Figure 4](#figure-4) falls).
- SmallThinker-4B-A0.6B's decode-phase energy per output token is the lowest of any model at short context, consistent with its top [tok/J](#appendix-i3) numbers in [Table 11](#table-11).

## 4. Backend Comparison: llama.cpp vs MLX-LM

Both backends ran identical Q4_K_M / 4-bit-MLX quantizations at single-user concurrency on the same Mac Mini M4. Three of the six models have data under both backends - see [caveats](#backend-caveats) for why the other three are llama.cpp-only.

### 4.1 Throughput and Efficiency Head-to-Head

At the **canonical cell** (ctx=30720, gen=1024):

<a id="table-15"></a>
**Table 15: llama.cpp vs MLX-LM, canonical cell**

| Model | Total Params | Active Params | llama.cpp <a href="#appendix-i1" style="color:inherit;text-decoration:none"><code>tok/s</code></a> | MLX-LM <a href="#appendix-i1" style="color:inherit;text-decoration:none"><code>tok/s</code></a> | LC ÷ MX | llama.cpp <a href="#appendix-i3" style="color:inherit;text-decoration:none"><code>tok/J</code></a> | MLX-LM <a href="#appendix-i3" style="color:inherit;text-decoration:none"><code>tok/J</code></a> | LC ÷ MX <a href="#appendix-i3" style="color:inherit;text-decoration:none"><code>tok/J</code></a> |
|-------|--------:|--------:|---------------:|-------------:|--------:|----------------:|-------------:|----------------:|
| Granite-4.0-H-Tiny   | 7B   | 1B   | 45.45 | **59.41** | 0.77× | 0.7829 | **0.9597** | 0.82× |
| LFM2.5-8B-A1B        | 8.3B | 1.5B | **63.73** | 54.99 | **1.16×** | **1.0251** | 0.9155 | **1.12×** |
| Trinity-Nano-Preview | 6B   | 1B   | **61.08** | 55.11 | **1.11×** | **1.2403** | 1.1992 | 1.03× |

### 4.2 Key Observations

**1. There is no universal winner between backends on Apple Silicon.**  
Unlike the Jetson CUDA case, where llama.cpp beat Ollama for almost every model, here the result is split: MLX-LM wins decisively for Granite-4.0-H-Tiny (1.3-1.4× faster across the whole sweep - see [Table 6](#table-6)), while llama.cpp wins narrowly for LFM2.5-8B-A1B and Trinity-Nano-Preview (5-16 % faster). This suggests the two frameworks' kernel optimisation priorities diverge by architecture (hybrid Mamba/attention vs. hybrid SSM/attention vs. standard MoE routing) rather than one framework being categorically faster on this hardware.

**2. MLX-LM draws less power for the same work.**  
Across all three shared models, MLX-LM's average total power (12.25-13.97 W) is consistently lower than llama.cpp's (13.44-14.93 W) - see [Table 7](#table-7) and [Table 8](#table-8). This is plausibly server/framework overhead rather than raw compute efficiency, since Granite's [tok/J](#appendix-i3) advantage under MLX-LM is larger than its power advantage alone would predict.

**3. Best-of-both-worlds recommendation.**

The resulting workload-specific choices are summarized in [Table 16](#table-16).

<a id="table-16"></a>
**Table 16: Recommended backend and model by use case**

| Use case | Backend | Model |
|----------|---------|-------|
| Max long-context throughput | **llama.cpp** | LFM2.5-8B-A1B |
| Max short-context throughput / efficiency | **llama.cpp** | SmallThinker-4B-A0.6B (no MLX weights published) |
| Granite family on Apple Silicon | **MLX-LM** | Granite-4.0-H-Tiny |
| Lowest average power for a given model | **MLX-LM** | any of the 3 shared models |

<a id="section-43"></a>
### 4.3 Data-Quality Caveat: MLX-LM Generation-Length Undershoot

`--ignore-eos` on llama.cpp forces every request to produce exactly the target `gen` token count; MLX-LM's server has no equivalent flag, so it stops as soon as the model emits an EOS token. For most models this causes only a small (<5 %) [OSL undershoot](#appendix-i12). **Granite-4.0-H-Tiny under MLX-LM is a severe outlier**: at gen=512 and gen=1024 targets, actual output length falls **40-72 % short** of target (e.g. 291.6 tokens actually generated against a 1024-token target). Under llama.cpp, the same model's [OSL](#appendix-i1) matches its target almost exactly (<3 % deviation) at every combo.

This means Granite-4.0-H-Tiny's MLX-LM [tok/s](#appendix-i1) and [tok/J](#appendix-i3) numbers at longer gen targets are computed over a *shorter-than-intended* generation window - a backend-specific early-stopping behaviour, not a hardware or power effect. Treat the MLX-LM Granite comparison in [Table 15](#table-15) as directionally correct but not perfectly apples-to-apples with the llama.cpp row for that model. Full per-cell [OSL mismatch](#appendix-i12) appears in [Figure I.2g](#figure-i2g) for llama.cpp and [Figure I.2g-b](#figure-i2g-b) for MLX-LM.

## 5. Conclusion

### What These Numbers Mean for Local Inference on Apple Silicon

Small MoE/hybrid-MoE inference on a $599 Mac Mini M4 16GB is practical well past the token counts most local-LLM benchmarks test. At the canonical 30,720-token-context cell:

- **LFM2.5-8B-A1B** sustains **63.7 [tok/s](#appendix-i1)** at 30k tokens of context - real-time-adjacent generation even with a full long document loaded
- **SmallThinker-4B-A0.6B** is the fastest model in the suite at any context under ~4k tokens (**130.75 [tok/s](#appendix-i1)** at ctx=256), and the single most [tok/J](#appendix-i3)-efficient cell in the entire sweep (**8.56** at ctx=256, gen=1024)
- **Every model** fits comfortably inside 16 GB unified memory, even at the 30k-token context, with 3-6 GB of headroom to spare
- **No model throttled** at any point across either backend's full sweep - the M4's cooling handles sustained MoE inference without a power-mode governor to fall back on

### The Two Practical Takeaways

1. **Match the model to your context length, not just its published benchmark numbers.** SmallThinker-4B-A0.6B's short-context throughput crown means little if your workload routinely sends 20k+ token prompts - at that length it is the *slowest* model in this suite, having lost more than half its speed. Check [retention](#appendix-i9) ([Table 12](#table-12)), not just peak [tok/s](#appendix-i1).
2. **Match the backend to the model, not a blanket "llama.cpp is faster" assumption.** That held on Jetson/CUDA; it does not hold cleanly here. Granite-4.0-H-Tiny specifically favors MLX-LM; benchmark your own model/backend pair before committing to one framework project-wide.

### What Is Not Yet Benchmarked

- **Multi-user concurrency**: all results are single-user (`--parallel 1` / `--decode-concurrency 1`). Real-world serving will see different throughput/latency profiles at concurrency > 1.
- **Neural Engine (ANE) offload**: neither backend under test routes any compute through the M4's 16-core ANE; an ANE-aware runtime could shift the power/throughput picture materially.
- **SmallThinker under MLX-LM**: no 4-bit MLX conversion is currently published for this model; if one becomes available, it is the most interesting backend comparison left to run given its short-context dominance under llama.cpp.
- **Gemma-4-E2B/E4B under MLX-LM**: blocked on `mlx_vlm.server` support for the native VLM weight structure - see [backend caveats](#backend-caveats).

---
<a id="appendix-a"></a>
## Appendix A: Full-Suite Comparison Dashboard (gen=1024)

This dashboard comprises the full-sweep view in [Figure A.1](#figure-a1), the combined throughput/efficiency view in [Figure A.2](#figure-a2), and the Pareto view in [Figure A.3](#figure-a3).

<a id="appendix-a1"></a>
<a id="figure-a1"></a>
**Figure A.1 - Full-sweep dashboard: [tok/s](#appendix-i1), [tok/J](#appendix-i3), latency, and power in one view, gen=1024 (llama.cpp, complete 6-model dataset)**

![Full dashboard gen=1024](artifacts/charts/i2a_dashboard_gen1024.png)

<a id="appendix-a2"></a>
<a id="figure-a2"></a>
**Figure A.2 - [Tok/s](#appendix-i1) + [Tok/J](#appendix-i3) combined, gen=1024, llama.cpp (solid) vs MLX-LM (dashed)**

![Tok/s and Tok/J combined](artifacts/charts/i2b_tok_s_tok_j_combined_gen1024.png)

<a id="appendix-a3"></a>
<a id="figure-a3"></a>
**Figure A.3 - [Tok/s](#appendix-i1) vs [Tok/J](#appendix-i3) Pareto scatter, gen=1024, llama.cpp (filled markers) vs MLX-LM (hollow markers)**

[Figure A.3](#figure-a3) places models that dominate on both axes toward the top-right.

![Tok/s vs Tok/J scatter](artifacts/charts/i2i_tok_s_vs_tok_j_scatter_gen1024.png)

---

<a id="appendix-b"></a>
## Appendix B: Latency and Power Breakdown (gen=1024)

Latency appears in [Figure B.1](#figure-b1); power is broken down by model in [Figure B.2](#figure-b2), by prompt length in [Figure B.3](#figure-b3), and at the canonical cell in [Figure B.4](#figure-b4).

<a id="appendix-b1"></a>
<a id="figure-b1"></a>
**Figure B.1 - Latency breakdown: [TTFT](#appendix-i11) / [ITL](#appendix-i11) / [RL](#appendix-i11) by model, gen=1024, llama.cpp (solid) vs MLX-LM (dashed)**

![Latency breakdown](artifacts/charts/i2c_latency_breakdown_gen1024.png)

<a id="appendix-b2"></a>
<a id="figure-b2"></a>
**Figure B.2 - Power breakdown: prefill vs [decode power](#appendix-i5) by model, gen=1024, llama.cpp (solid) vs MLX-LM (dashed)**

![Power breakdown](artifacts/charts/i2d_power_breakdown_gen1024.png)

<a id="appendix-b3"></a>
<a id="figure-b3"></a>
**Figure B.3 - Prefill vs [decode power](#appendix-i5) by prompt length, gen=1024, all prompts (llama.cpp, complete 6-model dataset)**

![Prefill vs decode power by prompt](artifacts/charts/i2k_prefill_decode_power_by_prompt.png)

<a id="appendix-b4"></a>
<a id="figure-b4"></a>
**Figure B.4 - Prefill vs [decode power](#appendix-i5), canonical cell (grouped bar), llama.cpp vs MLX-LM**

![Prefill vs decode power canonical](artifacts/charts/EP_prefill_decode_power_canonical.png)

---

<a id="appendix-c"></a>
## Appendix C: llama.cpp vs MLX-LM - Backend Ratio Detail

All ratios are llama.cpp ÷ MLX-LM. Values **> 1×** mean llama.cpp is faster / more efficient; **< 1×** means MLX-LM leads. Only the 3 models with both backends' data are shown - see [4.3](#section-43) for the Granite-4.0-H-Tiny [OSL](#appendix-i1)-undershoot caveat that affects its [tok/J](#appendix-i3) ratio at longer gen lengths.

Per-cell ratio range across the full 18-combo sweep is in [Table 6](#table-6); the dual-backend line/bar charts throughout sections 2-4 and Appendices A, B, D-H (llama.cpp solid vs MLX-LM dashed/hatched, same colour per model) provide the same underlying per-cell dataset used to build that table, viewable per model instead of pre-aggregated.

---

<a id="appendix-d"></a>
## Appendix D: Prefill / Decode / [Total Tok/J](#appendix-i6) - All Generation Lengths

Charts show all 6 llama.cpp models as solid colored lines and the 3 shared MLX-LM models as dashed lines in the same colour, across prompt lengths.

Prefill efficiency is shown in [Figures D.1a](#figure-d1a), [D.1b](#figure-d1b), and [D.1c](#figure-d1c); total efficiency is shown in [Figures D.2a](#figure-d2a), [D.2b](#figure-d2b), and [D.2c](#figure-d2c).

<a id="appendix-d1"></a>
### D.1 [Prefill tok/J](#appendix-i6) across generation lengths

<a id="figure-d1a"></a>
**Figure D.1a: [Prefill tok/J](#appendix-i6) vs prompt - gen=256**

![Prefill tok/J gen=256](artifacts/charts/22e_prefill_tokj_vs_prompt_gen256.png)

<a id="figure-d1b"></a>
**Figure D.1b: [Prefill tok/J](#appendix-i6) vs prompt - gen=512**

![Prefill tok/J gen=512](artifacts/charts/22e_prefill_tokj_vs_prompt_gen512.png)

<a id="figure-d1c"></a>
**Figure D.1c: [Prefill tok/J](#appendix-i6) vs prompt - gen=1024** *(canonical, also in [2.2](#section-22))*

![Prefill tok/J gen=1024](artifacts/charts/22e_prefill_tokj_vs_prompt_gen1024.png)

<a id="appendix-d2"></a>
### D.2 [Total tok/J](#appendix-i6) across generation lengths

<a id="figure-d2a"></a>
**Figure D.2a: [Total tok/J](#appendix-i6) vs prompt - gen=256**

![Total tok/J gen=256](artifacts/charts/22g_total_tokj_vs_prompt_gen256.png)

<a id="figure-d2b"></a>
**Figure D.2b: [Total tok/J](#appendix-i6) vs prompt - gen=512**

![Total tok/J gen=512](artifacts/charts/22g_total_tokj_vs_prompt_gen512.png)

<a id="figure-d2c"></a>
**Figure D.2c: [Total tok/J](#appendix-i6) vs prompt - gen=1024** *(canonical, also in [2.2](#section-22))*

![Total tok/J gen=1024](artifacts/charts/22g_total_tokj_vs_prompt_gen1024.png)

---

<a id="appendix-e"></a>
## Appendix E: [Request Latency](#appendix-i11) (E2E) - All Generation Lengths

[Request latency](#appendix-i11) by generation length appears in [Figures E.1a](#figure-e1a), [E.1b](#figure-e1b), and [E.1c](#figure-e1c); percentile spread appears in [Figure E.2](#figure-e2).

<a id="appendix-e1"></a>
### E.1 [Request latency](#appendix-i11) vs prompt length (by gen length), llama.cpp (solid) vs MLX-LM (dashed)

<a id="figure-e1a"></a>
**Figure E.1a: [Request latency](#appendix-i11) vs prompt - gen=256**

![RL vs prompt gen=256](artifacts/charts/EA_request_latency_vs_prompt_gen256.png)

<a id="figure-e1b"></a>
**Figure E.1b: [Request latency](#appendix-i11) vs prompt - gen=512**

![RL vs prompt gen=512](artifacts/charts/EA_request_latency_vs_prompt_gen512.png)

<a id="figure-e1c"></a>
**Figure E.1c: [Request latency](#appendix-i11) vs prompt - gen=1024** *(canonical)*

![RL vs prompt gen=1024](artifacts/charts/EA_request_latency_vs_prompt_gen1024.png)

<a id="figure-e2"></a>
**Figure E.2: [Request latency](#appendix-i11) p50/p90/p99 spread at the canonical cell, llama.cpp (solid edge) vs MLX-LM (hatched)**

![RL percentile spread](artifacts/charts/i2j_request_latency_percentiles_ctx30720_gen1024.png)

---

<a id="appendix-f"></a>
## Appendix F: [TTFT](#appendix-i11) - All Combinations

[TTFT](#appendix-i11) by prompt length appears in [Figures F.1a](#figure-f1a) and [F.1b](#figure-f1b); the backend heatmaps appear in [Figures F.2](#figure-f2) and [F.2b](#figure-f2b).

<a id="appendix-f1"></a>
### F.1 [TTFT](#appendix-i11) vs prompt length (by gen length), llama.cpp (solid) vs MLX-LM (dashed)

<a id="figure-f1a"></a>
**Figure F.1a: [TTFT](#appendix-i11) vs prompt - gen=256**

![TTFT vs prompt gen=256](artifacts/charts/EG_ttft_vs_prompt_gen256.png)

<a id="figure-f1b"></a>
**Figure F.1b: [TTFT](#appendix-i11) vs prompt - gen=1024** *(canonical)*

![TTFT vs prompt gen=1024](artifacts/charts/EG_ttft_vs_prompt_gen1024.png)

<a id="figure-f2"></a>
**Figure F.2: [TTFT](#appendix-i11) heatmap, all models × all prompt/gen combos, llama.cpp**

![TTFT heatmap](artifacts/charts/EG_ttft_heatmap_all_models.png)

<a id="figure-f2b"></a>
**Figure F.2b: [TTFT](#appendix-i11) heatmap, MLX-LM (3 shared models)**

![TTFT heatmap MLX-LM](artifacts/charts/EG_ttft_heatmap_mlxlm.png)

---

<a id="appendix-g"></a>
## Appendix G: [ITL](#appendix-i11) - All Combinations

[ITL](#appendix-i11) by prompt length appears in [Figures G.1a](#figure-g1a), [G.1b](#figure-g1b), and [G.1c](#figure-g1c); the backend heatmaps appear in [Figures G.2](#figure-g2) and [G.2b](#figure-g2b).

<a id="appendix-g1"></a>
### G.1 [ITL](#appendix-i11) vs prompt length (by gen length), llama.cpp (solid) vs MLX-LM (dashed)

<a id="figure-g1a"></a>
**Figure G.1a: [ITL](#appendix-i11) vs prompt - gen=256**

![ITL vs prompt gen=256](artifacts/charts/EH_itl_vs_prompt_gen256.png)

<a id="figure-g1b"></a>
**Figure G.1b: [ITL](#appendix-i11) vs prompt - gen=512**

![ITL vs prompt gen=512](artifacts/charts/EH_itl_vs_prompt_gen512.png)

<a id="figure-g1c"></a>
**Figure G.1c: [ITL](#appendix-i11) vs prompt - gen=1024** *(canonical)*

![ITL vs prompt gen=1024](artifacts/charts/EH_itl_vs_prompt_gen1024.png)

<a id="figure-g2"></a>
**Figure G.2: [ITL](#appendix-i11) heatmap, all models × all prompt/gen combos, llama.cpp**

![ITL heatmap](artifacts/charts/EH_itl_heatmap_all_models.png)

<a id="figure-g2b"></a>
**Figure G.2b: [ITL](#appendix-i11) heatmap, MLX-LM (3 shared models)**

![ITL heatmap MLX-LM](artifacts/charts/EH_itl_heatmap_mlxlm.png)

---

<a id="appendix-h"></a>
## Appendix H: [Prefill Throughput](#appendix-i8) - All Combinations

[Prefill throughput](#appendix-i8) by prompt length appears in [Figures H.1a](#figure-h1a) and [H.1b](#figure-h1b), the all-generation view in [Figure H.2](#figure-h2), and the backend heatmaps in [Figures H.3](#figure-h3) and [H.3b](#figure-h3b).

<a id="appendix-h1"></a>
### H.1 [Prefill throughput](#appendix-i8) vs prompt length (by gen length), llama.cpp (solid) vs MLX-LM (dashed)

<a id="figure-h1a"></a>
**Figure H.1a: [Prefill tok/s](#appendix-i8) vs prompt - gen=256**

![Prefill tput vs prompt gen=256](artifacts/charts/EI_prefill_tput_vs_prompt_gen256.png)

<a id="figure-h1b"></a>
**Figure H.1b: [Prefill tok/s](#appendix-i8) vs prompt - gen=1024** *(canonical, same chart as [Figure 2](#figure-2) in [2.1](#section-21); [2.4](#section-24) has the avg-over-all-prompts bar view instead)*

![Prefill tput vs prompt gen=1024](artifacts/charts/EI_prefill_tput_vs_prompt_gen1024.png)

<a id="figure-h2"></a>
**Figure H.2: [Prefill throughput](#appendix-i8) across all generation lengths (line chart), llama.cpp (solid) vs MLX-LM (dashed)**

![Prefill TPS all gen](artifacts/charts/i2e_prefill_tps_all_gen.png)

<a id="figure-h3"></a>
**Figure H.3: [Prefill throughput](#appendix-i8) heatmap, all models × all prompt/gen combos, llama.cpp**

![Prefill tput heatmap](artifacts/charts/EI_prefill_tput_heatmap_all_models.png)

<a id="figure-h3b"></a>
**Figure H.3b: [Prefill throughput](#appendix-i8) heatmap, MLX-LM (3 shared models)**

![Prefill tput heatmap MLX-LM](artifacts/charts/EI_prefill_tput_heatmap_mlxlm.png)

---

<a id="appendix-i2-charts"></a>
## Appendix I.2: Extended Dashboard Charts

Supplementary charts from `generate_appendix_i2_charts.py`, covering the full 18-combo × 6-model llama.cpp dataset from angles not already shown above, each paired with an MLX-LM companion for the 3 shared models where the metric allows a heatmap split (a single heatmap cell can only hold one number, so backend pairs are two adjacent grids rather than one merged one - see the line/point convention note at the top of [section 2](#section-2)).

- [Output tok/s](#appendix-i1): [Figure I.2a](#figure-i2a) and [Figure I.2a-b](#figure-i2a-b)
- [Output tok/J](#appendix-i3): [Figure I.2b](#figure-i2b) and [Figure I.2b-b](#figure-i2b-b)
- [Prefill power](#appendix-i5): [Figure I.2c](#figure-i2c) and [Figure I.2c-b](#figure-i2c-b)
- [Decode power](#appendix-i5): [Figure I.2d](#figure-i2d) and [Figure I.2d-b](#figure-i2d-b)
- [Tok/J](#appendix-i3) by generation length and [peak RAM](#appendix-i13): [Figure I.2e](#figure-i2e) and [Figure I.2f](#figure-i2f)
- [OSL mismatch](#appendix-i12): [Figure I.2g](#figure-i2g) and [Figure I.2g-b](#figure-i2g-b)
- [Request latency](#appendix-i11): [Figure I.2h](#figure-i2h)

<a id="figure-i2a"></a>
**Figure I.2a: [Output tok/s](#appendix-i1) heatmap, all models × all combos, llama.cpp**

![Tok/s heatmap](artifacts/charts/E_tok_s_heatmap_all_models.png)

<a id="figure-i2a-b"></a>
**Figure I.2a-b: [Output tok/s](#appendix-i1) heatmap, MLX-LM (3 shared models)**

![Tok/s heatmap MLX-LM](artifacts/charts/E_tok_s_heatmap_mlxlm.png)

<a id="figure-i2b"></a>
**Figure I.2b: [Output tok/J](#appendix-i3) heatmap, all models × all combos, llama.cpp**

![Tok/J heatmap](artifacts/charts/7_tok_j_heatmap_all_models.png)

<a id="figure-i2b-b"></a>
**Figure I.2b-b: [Output tok/J](#appendix-i3) heatmap, MLX-LM (3 shared models)**

![Tok/J heatmap MLX-LM](artifacts/charts/7_tok_j_heatmap_mlxlm.png)

<a id="figure-i2c"></a>
**Figure I.2c: [Prefill power](#appendix-i5) heatmap, all models × all combos, llama.cpp**

![Prefill power heatmap](artifacts/charts/EP_prefill_power_heatmap_all_models.png)

<a id="figure-i2c-b"></a>
**Figure I.2c-b: [Prefill power](#appendix-i5) heatmap, MLX-LM (3 shared models)**

![Prefill power heatmap MLX-LM](artifacts/charts/EP_prefill_power_heatmap_mlxlm.png)

<a id="figure-i2d"></a>
**Figure I.2d: [Decode power](#appendix-i5) heatmap, all models × all combos, llama.cpp**

![Decode power heatmap](artifacts/charts/EP_decode_power_heatmap_all_models.png)

<a id="figure-i2d-b"></a>
**Figure I.2d-b: [Decode power](#appendix-i5) heatmap, MLX-LM (3 shared models)**

![Decode power heatmap MLX-LM](artifacts/charts/EP_decode_power_heatmap_mlxlm.png)

<a id="figure-i2e"></a>
**Figure I.2e: [Tok/J](#appendix-i3) grouped bars by generation length (llama.cpp, complete 6-model dataset)**

![Tok/J grouped bars](artifacts/charts/i2h_tok_j_grouped_bars.png)

<a id="figure-i2f"></a>
**Figure I.2f: [Peak RAM](#appendix-i13) ([RSS](#appendix-i13)) by model, llama.cpp vs MLX-LM (max across all combos each backend ran)**

![Peak RAM](artifacts/charts/i2g_peak_ram.png)

<a id="figure-i2g"></a>
**Figure I.2g: [OSL mismatch](#appendix-i12) heatmap - actual vs target output tokens, all models × all combos, llama.cpp**

[Figure I.2g](#figure-i2g) shows that every llama.cpp cell stays within a few percent of target thanks to `--ignore-eos`.

![OSL mismatch heatmaps](artifacts/charts/i2l_osl_mismatch_heatmaps.png)

<a id="figure-i2g-b"></a>
**Figure I.2g-b: [OSL mismatch](#appendix-i12) heatmap, MLX-LM (3 shared models)**

[Figure I.2g-b](#figure-i2g-b) confirms the [section 4.3](#section-43) MLX-LM Granite undershoot directly: -70.7 to -72.7 % at gen=1024 and -41.3 to -45.3 % at gen=512, vs single-digit-percent deviation for lfm2.5-8b-a1b and trinity-nano at every combo.

![OSL mismatch heatmaps MLX-LM](artifacts/charts/i2l_osl_mismatch_heatmaps_mlxlm.png)

<a id="figure-i2h"></a>
**Figure I.2h: [Request latency](#appendix-i11) heatmaps, all models × all combos (llama.cpp, complete 6-model dataset)**

![Request latency heatmaps](artifacts/charts/i2f_request_latency_heatmaps.png)

---

<a id="appendix-i"></a>
## Appendix I: All Metrics, Formulas, and Calculation Methods

This appendix documents every metric reported in this benchmark, its formula, its source, and any caveats.

<a id="glossary"></a>
<a id="appendix-i1"></a>
### I.1 Raw inputs from aiperf and powermetrics

[Table I.1](#table-i1) defines the raw inputs used by the formulas in this appendix.

<a id="table-i1"></a>
**Table I.1: Raw metric inputs, sources, and definitions**

| Symbol | Source | Definition |
|--------|--------|------------|
| `ISL` | aiperf JSON `input_sequence_length.p50` | Actual input tokens processed per request (may differ from target due to tokenizer rounding) |
| `OSL` | aiperf JSON `output_sequence_length.p50` | Actual output tokens generated per request - see [4.3](#section-43) for the MLX-LM undershoot caveat |
| `TTFT` | aiperf JSON `time_to_first_token.p50` (ms) | Median time from request sent to first output token received; proxy for prefill duration |
| `ITL` | aiperf JSON `inter_token_latency.p50` (ms) | Median time between consecutive output tokens; per-token decode cost |
| `RL` | aiperf JSON `request_latency.p50` (ms) | Median total wall time per request: TTFT + all inter-token intervals |
| `tok_s` | aiperf JSON `output_token_throughput_per_user.p50` | Output tokens per second, single-user |
| `prefill_tput` | aiperf JSON `prefill_throughput_per_user.p50` | Input tokens processed per second during prefill phase |
| `p50_decode_s` | computed | Median decode duration in seconds, from per-request timestamps in `profile_export.jsonl`, or `(RL_p50 - TTFT_p50) / 1000` as fallback |
| `t0`, `t1` | aiperf JSON `start_time`, `end_time` (ISO 8601) | Wall-clock start and end of the full 20-request profiling run |
| `mW_i` | `powermetrics` `Combined Power (CPU + GPU + ANE)` field (mW) | Instantaneous combined-rail power at sample `i` |

---

<a id="appendix-i2"></a>
### I.2 Power

```
avg_power_W = mean(mW_i for all powermetrics samples in the combo window) / 1000
```

- `Combined Power (CPU + GPU + ANE)` is macOS's own accounting of the three on-package power domains that matter for inference; it does not include board-level overhead (storage, USB, display) the way Jetson's `VDD_IN` differs from `VDD_CPU_GPU_CV`.
- `powermetrics` interval: 50 ms.

---

<a id="appendix-i3"></a>
### I.3 Output tok/J (main efficiency metric)

```
output_tok_J = OSL / (decode_power_W * p50_decode_s)
```

Higher is better. Output tokens generated per joule of **decode-phase** energy only - the primary metric of this benchmark, and the one used throughout section 2-4.

---

<a id="appendix-i4"></a>
### I.4 Request latency energy

```
total_J = avg_power_W * (RL / 1000)
```

Energy consumed by one average request from first byte sent to last token received.

---

<a id="appendix-i5"></a>
### I.5 Prefill and decode energy

```
prefill_J  = prefill_power_W * (TTFT / 1000)
decode_J   = decode_power_W  * p50_decode_s
total_J    = prefill_J + decode_J

prefill_%  = prefill_J / total_J * 100
```

Prefill/decode windows come from `profile_export.jsonl` per-request timestamps; falls back to a TTFT/RL timeline reconstruction if the file is absent for a combo (see `compute_phase_power` in `generate_combined_charts.py`).

---

<a id="appendix-i6"></a>
### I.6 Phase tok/J metrics

```
prefill_tok_J = ISL / prefill_J
decode_tok_J  = OSL / decode_J        (== output_tok_J above)
total_tok_J   = (ISL + OSL) / total_J
```

- `prefill_tok_J`: input tokens processed per joule of prefill energy.
- `decode_tok_J`: output tokens generated per joule of decode energy - identical metric to I.3.
- `total_tok_J`: all tokens (in + out) per joule of total request energy.

---

<a id="appendix-i7"></a>
### I.7 mJ per output token

```
mJ_per_output_tok = (decode_J / OSL) * 1000
                  = 1000 / decode_tok_J
```

Millijoules per generated output token.

---

<a id="appendix-i8"></a>
### I.8 Prefill throughput

```
prefill_tput (tok/s) = aiperf JSON prefill_throughput_per_user.p50
```

Directly from aiperf. Scales with prompt length (longer prompts hit peak GPU utilisation).

---

<a id="appendix-i9"></a>
### I.9 Context-length retention ([Table 12](#table-12) / [Figure 7](#figure-7))

```
retention_% = tok_s(ctx=30720, gen=1024) / tok_s(ctx=256, gen=1024) * 100
```

Computed per model, llama.cpp only (MLX-LM has 30720-context data for all 3 shared models too, but the headline retention comparison in [3.2](#section-32) uses the complete 6-model llama.cpp set).

---

<a id="appendix-i10"></a>
### I.10 Best total tok/J per model

```
best_total_tok_J(model) = max(total_tok_J(gen, ctx))
                          over all gen in {256, 512, 1024}
                          and all ctx in {256, 512, 1024, 2048, 4096, 30720}
```

The single highest total tok/J value observed for that model across all 18 combinations.

---

<a id="appendix-i11"></a>
### I.11 TTFT, ITL, RL percentiles

All percentile variants come directly from aiperf JSON without further computation:

```
TTFT       = time_to_first_token.p50   (canonical; p50 used everywhere)
TTFT_p90   = time_to_first_token.p90
TTFT_p99   = time_to_first_token.p99
ITL        = inter_token_latency.p50    (canonical; p50 used everywhere)
ITL_p99    = inter_token_latency.p99
RL         = request_latency.p50        (canonical; p50 used everywhere)
RL_p99     = request_latency.p99
```

---

<a id="appendix-i12"></a>
### I.12 OSL mismatch

```
OSL_mismatch_% = (OSL_actual - OSL_target) / OSL_target * 100
```

Reported per combo in `report.md`. Near-zero for every llama.cpp cell (`--ignore-eos` forces the target length); can be strongly negative for MLX-LM, especially Granite-4.0-H-Tiny at longer gen targets - see [4.3](#section-43).

---

<a id="appendix-i13"></a>
### I.13 Peak RAM (RSS)

```
peak_ram_MB = max(rss_sample for all 50ms samples in the combo window)
```

Sampled via `ps -o rss=` on the server process PID. Captures model weights + KV-cache + activations at that specific prompt/gen length - grows with context, not a fixed "model size" figure.

---

## Power Measurement Methodology Note

All **tok/J** and power figures in this report use the **Combined (CPU + GPU + ANE)** rail from macOS `powermetrics` at 50 ms intervals.

> **Method:** `tok/J = OSL ÷ (decode_power_W × ` [<code>p50_decode_s</code>](#appendix-i1)`)` - output tokens per joule of decode-phase energy.
>
> **Both backends:** per-request decode-phase power extracted from each combo's own `powermetrics.log` using `profile_export.jsonl` timestamps, falling back to a TTFT/RL timeline reconstruction when the timestamped log is unavailable for a cell.
>
> **No prefill-power blind spot:** unlike the Jetson `tegrastats` case (500 ms interval, missed short prefills), `powermetrics` here samples every 50 ms - short prefill windows at low context still get several samples, so the [I.12](#appendix-i12)-style "prefill approximated" caveat that applied to the Jetson benchmark does not apply to this dataset.
