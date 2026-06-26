#!/usr/bin/env python3
"""Generate Appendix A line charts (per-metric, both backends) and Appendix B thermal scatter.

Appendix A: one line chart per metric column (tok/s, TTFT, ITL, Power, tok/J),
            each chart shows all 8 models × 2 backends (llamacpp solid, ollama dashed)
            across the 4 power modes on the x-axis.

Appendix B: scatter plot of Avg Power (W) vs Peak TJ (°C), colored by mode,
            shaped by backend (llamacpp vs ollama), with model labels.
"""

import json, re, statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

ROOT      = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
OUT_DIR   = ARTIFACTS / "charts"
OUT_DIR.mkdir(exist_ok=True)

# ── directory mapping ──────────────────────────────────────────────────────────
LLAMACPP_DIRS = {
    "7W":   "llamacpp-hf-7w",
    "15W":  "llamacpp-hf-15w",
    "25W":  "llamacpp-hf-25w",
    "MAXN": "llamacpp-hf-maxn",
}
OLLAMA_DIRS = {
    "7W":   "blog-all-20260607-0403-7w",
    "15W":  "blog-all-20260606-0139-15w",
    "25W":  "blog-all-20260622-0159-25w",
    "MAXN": "blog-all-20260621-1401-maxn",
}

MODELS = [
    "smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "lfm2.5-350m",
    "lfm2.5-1.2b", "qwen3-0.6b", "llama3.2-1b", "gemma3-1b",
]
MODEL_LABELS = {
    "smollm2-135m": "SmolLM2-135M",
    "smollm2-360m": "SmolLM2-360M",
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "lfm2.5-350m":  "LFM2.5-350M",
    "lfm2.5-1.2b":  "LFM2.5-1.2B",
    "qwen3-0.6b":   "Qwen3-0.6B",
    "llama3.2-1b":  "Llama3.2-1B",
    "gemma3-1b":    "Gemma3-1B",
}
MODES = ["7W", "15W", "25W", "MAXN"]

# ── hardcoded power and tok/J from the report tables (verified against raw data) ──
# > Method: tok/J = OSL ÷ (decode_power_W × p50_decode_s).
# > Ollama: power extracted from per-mode tegrastats.log via model_timing.log
# >   windows (per-request decode-phase).
# > llama.cpp 7W: whole-run average power (no tegrastats retained for that run).
# > llama.cpp 15W/25W/MAXN: per-request decode-phase tegrastats.
LLAMACPP_POWER = {
    "7W":   {"smollm2-135m":1.99,"smollm2-360m":2.27,"qwen2.5-0.5b":2.22,"lfm2.5-350m":2.10,"lfm2.5-1.2b":2.34,"qwen3-0.6b":1.98,"llama3.2-1b":2.26,"gemma3-1b":1.96},
    "15W":  {"smollm2-135m":4.21,"smollm2-360m":4.90,"qwen2.5-0.5b":5.28,"lfm2.5-350m":4.95,"lfm2.5-1.2b":6.00,"qwen3-0.6b":4.94,"llama3.2-1b":6.06,"gemma3-1b":4.98},
    "25W":  {"smollm2-135m":5.64,"smollm2-360m":6.69,"qwen2.5-0.5b":7.05,"lfm2.5-350m":6.79,"lfm2.5-1.2b":8.52,"qwen3-0.6b":6.77,"llama3.2-1b":8.60,"gemma3-1b":6.82},
    "MAXN": {"smollm2-135m":6.51,"smollm2-360m":7.22,"qwen2.5-0.5b":8.67,"lfm2.5-350m":7.87,"lfm2.5-1.2b":9.85,"qwen3-0.6b":8.11,"llama3.2-1b":10.65,"gemma3-1b":8.56},
}
LLAMACPP_TOKJ = {
    "7W":   {"smollm2-135m":21.7,"smollm2-360m":11.0,"qwen2.5-0.5b":7.3,"lfm2.5-350m":11.8,"lfm2.5-1.2b":4.8,"qwen3-0.6b":6.2,"llama3.2-1b":4.5,"gemma3-1b":4.9},
    "15W":  {"smollm2-135m":27.58,"smollm2-360m":14.71,"qwen2.5-0.5b":13.14,"lfm2.5-350m":16.19,"lfm2.5-1.2b":6.18,"qwen3-0.6b":6.83,"llama3.2-1b":5.34,"gemma3-1b":5.66},
    "25W":  {"smollm2-135m":29.62,"smollm2-360m":15.50,"qwen2.5-0.5b":13.30,"lfm2.5-350m":17.16,"lfm2.5-1.2b":6.37,"qwen3-0.6b":7.26,"llama3.2-1b":5.48,"gemma3-1b":6.02},
    "MAXN": {"smollm2-135m":24.72,"smollm2-360m":12.64,"qwen2.5-0.5b":11.78,"lfm2.5-350m":14.53,"lfm2.5-1.2b":5.37,"qwen3-0.6b":6.68,"llama3.2-1b":4.89,"gemma3-1b":5.19},
}
OLLAMA_POWER = {
    "7W":   {"smollm2-135m":1.90,"smollm2-360m":2.10,"qwen2.5-0.5b":2.01,"lfm2.5-350m":1.78,"lfm2.5-1.2b":2.06,"qwen3-0.6b":2.10,"llama3.2-1b":2.33,"gemma3-1b":2.05},
    "15W":  {"smollm2-135m":4.18,"smollm2-360m":4.68,"qwen2.5-0.5b":4.53,"lfm2.5-350m":3.79,"lfm2.5-1.2b":4.64,"qwen3-0.6b":4.79,"llama3.2-1b":5.81,"gemma3-1b":4.76},
    "25W":  {"smollm2-135m":5.68,"smollm2-360m":6.59,"qwen2.5-0.5b":6.22,"lfm2.5-350m":4.37,"lfm2.5-1.2b":5.56,"qwen3-0.6b":6.69,"llama3.2-1b":8.32,"gemma3-1b":6.59},
    "MAXN": {"smollm2-135m":7.17,"smollm2-360m":8.14,"qwen2.5-0.5b":7.82,"lfm2.5-350m":5.58,"lfm2.5-1.2b":6.91,"qwen3-0.6b":8.39,"llama3.2-1b":10.56,"gemma3-1b":8.03},
}
OLLAMA_TOKJ = {
    "7W":   {"smollm2-135m":19.21,"smollm2-360m":10.01,"qwen2.5-0.5b":7.80,"lfm2.5-350m":9.02,"lfm2.5-1.2b":4.77,"qwen3-0.6b":4.10,"llama3.2-1b":5.06,"gemma3-1b":4.76},
    "15W":  {"smollm2-135m":20.14,"smollm2-360m":10.41,"qwen2.5-0.5b":8.70,"lfm2.5-350m":6.64,"lfm2.5-1.2b":4.10,"qwen3-0.6b":6.78,"llama3.2-1b":5.38,"gemma3-1b":5.10},
    "25W":  {"smollm2-135m":21.26,"smollm2-360m":10.61,"qwen2.5-0.5b":8.95,"lfm2.5-350m":6.39,"lfm2.5-1.2b":3.94,"qwen3-0.6b":6.99,"llama3.2-1b":5.38,"gemma3-1b":5.29},
    "MAXN": {"smollm2-135m":18.65,"smollm2-360m":9.46,"qwen2.5-0.5b":7.84,"lfm2.5-350m":5.49,"lfm2.5-1.2b":3.43,"qwen3-0.6b":6.09,"llama3.2-1b":4.73,"gemma3-1b":4.46},
}

# llamacpp thermal from Table 16 (avg power W, peak TJ °C per mode per model)
LLAMACPP_THERMAL = {
    "7W":  {"smollm2-135m":(1.99,50.3),"smollm2-360m":(2.27,52.1),"qwen2.5-0.5b":(2.22,52.2),"lfm2.5-350m":(2.10,53.0),"lfm2.5-1.2b":(2.34,54.0),"qwen3-0.6b":(1.98,47.5),"llama3.2-1b":(2.26,47.6),"gemma3-1b":(1.96,50.5)},
    "15W": {"smollm2-135m":(4.16,60.2),"smollm2-360m":(4.85,63.3),"qwen2.5-0.5b":(5.22,59.6),"lfm2.5-350m":(4.95,62.1),"lfm2.5-1.2b":(6.06,65.5),"qwen3-0.6b":(4.98,65.4),"llama3.2-1b":(6.06,65.7),"gemma3-1b":(4.98,63.6)},
    "25W": {"smollm2-135m":(5.54,54.3),"smollm2-360m":(6.66,58.6),"qwen2.5-0.5b":(6.96,59.1),"lfm2.5-350m":(6.79,58.1),"lfm2.5-1.2b":(8.49,63.0),"qwen3-0.6b":(6.81,63.1),"llama3.2-1b":(8.58,66.1),"gemma3-1b":(6.83,61.9)},
    "MAXN":{"smollm2-135m":(6.36,53.2),"smollm2-360m":(7.23,56.8),"qwen2.5-0.5b":(8.59,62.8),"lfm2.5-350m":(7.85,56.8),"lfm2.5-1.2b":(9.72,63.5),"qwen3-0.6b":(8.38,64.0),"llama3.2-1b":(10.64,69.5),"gemma3-1b":(8.52,67.0)},
}

# ── model colour palette ───────────────────────────────────────────────────────
MODEL_COLORS = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))

sns.set_theme(style="darkgrid", font_scale=1.05)


# ── helpers ────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def load_aiperf_p50(art_dir, model, backend_subdir=""):
    """Return (tok_s, ttft_ms, itl_ms) p50 from canonical cell profile_export_aiperf.json."""
    if backend_subdir:
        p = ARTIFACTS / art_dir / backend_subdir / model / "gen256" / "ctx2048" / "profile_export_aiperf.json"
    else:
        p = ARTIFACTS / art_dir / model / "gen256" / "ctx2048" / "profile_export_aiperf.json"
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    def g(k):
        return (d.get(k) or {}).get("p50")
    return g("output_token_throughput_per_user"), g("time_to_first_token"), g("inter_token_latency")


def parse_tegrastats_tj(art_dir):
    """Parse tegrastats.log; return list of (unix_ts, vdd_cpu_gpu_cv_mw, tj_c)."""
    p = ARTIFACTS / art_dir / "tegrastats.log"
    records = []
    if not p.exists():
        return records
    with open(p) as f:
        for line in f:
            m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), '%m-%d-%Y %H:%M:%S').timestamp()
            pw = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
            tj = re.search(r'tj@([\d.]+)C', line)
            if pw and tj:
                records.append((ts, int(pw.group(1)), float(tj.group(1))))
    return records


def model_windows(art_dir):
    """Parse model_timing.log; return {model_name: (start_unix, end_unix)}."""
    p = ARTIFACTS / art_dir / "model_timing.log"
    windows = {}
    starts = {}
    if not p.exists():
        return windows
    with open(p) as f:
        for line in f:
            line = line.strip()
            m = re.match(r'MODEL_START:ollama:([\w.\-]+):[\w]+:(\d+)', line)
            if m:
                starts[m.group(1)] = int(m.group(2))
                continue
            m = re.match(r'MODEL_END:ollama:([\w.\-]+):[\w]+:(\d+)', line)
            if m:
                model = m.group(1)
                if model in starts:
                    windows[model] = (starts[model], int(m.group(2)))
    return windows


def ollama_thermal_for_mode(art_dir):
    """Return {model: (avg_power_w, peak_tj_c)} for an ollama run directory."""
    tegra = parse_tegrastats_tj(art_dir)
    windows = model_windows(art_dir)
    result = {}
    for model, (t0, t1) in windows.items():
        if model not in MODELS:
            continue
        in_window = [(mw, tj) for (ts, mw, tj) in tegra if t0 <= ts <= t1]
        if not in_window:
            continue
        avg_pw = statistics.median(mw for mw, _ in in_window) / 1000.0
        peak_tj = max(tj for _, tj in in_window)
        result[model] = (avg_pw, peak_tj)
    return result


# ── extract latency data (tok/s, TTFT, ITL) from JSONs ───────────────────────

print("Loading aiperf data from JSON files...")
llamacpp_toks  = {}
llamacpp_ttfts = {}
llamacpp_itls  = {}
ollama_toks    = {}
ollama_ttfts   = {}
ollama_itls    = {}

for mode in MODES:
    llamacpp_toks[mode]  = {}
    llamacpp_ttfts[mode] = {}
    llamacpp_itls[mode]  = {}
    ollama_toks[mode]    = {}
    ollama_ttfts[mode]   = {}
    ollama_itls[mode]    = {}
    for model in MODELS:
        ts, tt, it = load_aiperf_p50(LLAMACPP_DIRS[mode], model, "")
        llamacpp_toks[mode][model]  = ts
        llamacpp_ttfts[mode][model] = tt
        llamacpp_itls[mode][model]  = it

        ts, tt, it = load_aiperf_p50(OLLAMA_DIRS[mode], model, "ollama")
        ollama_toks[mode][model]  = ts
        ollama_ttfts[mode][model] = tt
        ollama_itls[mode][model]  = it


# ── extract ollama thermal data ────────────────────────────────────────────────

print("Extracting ollama thermal data from tegrastats...")
OLLAMA_THERMAL = {}
for mode, art_dir in OLLAMA_DIRS.items():
    OLLAMA_THERMAL[mode] = ollama_thermal_for_mode(art_dir)


# ── APPENDIX A: per-metric line charts ────────────────────────────────────────

METRICS = [
    ("tok_s",  "Output Tok/s",    "tokens / second"),
    ("ttft",   "TTFT p50",        "ms"),
    ("itl",    "ITL p50",         "ms"),
    ("power",  "Decode Power",    "W"),
    ("tokj",   "Output Tok/J",    "tok / J"),
]

DATA = {
    "tok_s":  (llamacpp_toks,    ollama_toks),
    "ttft":   (llamacpp_ttfts,   ollama_ttfts),
    "itl":    (llamacpp_itls,    ollama_itls),
    "power":  (LLAMACPP_POWER,   OLLAMA_POWER),
    "tokj":   (LLAMACPP_TOKJ,    OLLAMA_TOKJ),
}

X = np.arange(len(MODES))
MODE_XLAB = ["7W", "15W", "25W", "MAXN"]

for metric_key, metric_title, y_label in METRICS:
    lc_data, ol_data = DATA[metric_key]

    fig, ax = plt.subplots(figsize=(13, 7))

    for model in MODELS:
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]

        lc_vals = [lc_data[m].get(model) for m in MODES]
        ol_vals = [ol_data[m].get(model) for m in MODES]

        lc_x = [X[i] for i, v in enumerate(lc_vals) if v is not None]
        lc_y = [v for v in lc_vals if v is not None]
        ol_x = [X[i] for i, v in enumerate(ol_vals) if v is not None]
        ol_y = [v for v in ol_vals if v is not None]

        if lc_y:
            ax.plot(lc_x, lc_y, color=color, linestyle="-", linewidth=2.5,
                    marker="o", markersize=8)
        if ol_y:
            ax.plot(ol_x, ol_y, color=color, linestyle="--", linewidth=2.0,
                    marker="s", markersize=7, alpha=0.8)

    ax.set_xticks(X)
    ax.set_xticklabels(MODE_XLAB, fontsize=13)
    ax.set_xlabel("Power Mode", fontsize=13)
    ax.set_ylabel(f"{metric_title} ({y_label})", fontsize=13)
    ax.set_title(f"{metric_title} — llama.cpp vs Ollama, ctx=2048 gen=256",
                 fontsize=14, pad=10)
    ax.tick_params(axis="y", labelsize=11)

    model_handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=2.5, marker="o",
                      markersize=6, label=MODEL_LABELS[m])
        for m in MODELS
    ]
    spacer = mlines.Line2D([], [], color="none", label="")
    style_handles = [
        mlines.Line2D([], [], color="gray", linestyle="-",  linewidth=2.5,
                      marker="o", markersize=6, label="llama.cpp"),
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2.0,
                      marker="s", markersize=6, alpha=0.8, label="Ollama"),
    ]
    ax.legend(
        handles=model_handles + [spacer] + style_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=10,
        frameon=True,
        ncol=1,
        title="Model  /  Backend",
        title_fontsize=10,
        handlelength=2.2,
        handletextpad=0.7,
    )

    fig.tight_layout()
    save(fig, f"appendix_a_{metric_key}.png")


# ── APPENDIX B: two stacked panels, shared x, shared legend ──────────────────
# Top: Avg Power (W). Bottom: Peak TJ (°C).
# Each panel = same size as the Appendix A charts. Clean, no twinx confusion.

print("Generating Appendix B thermal chart...")

X_B    = np.arange(len(MODES))
MXLAB  = ["7W", "15W", "25W", "MAXN"]

fig, (ax_pw, ax_tj) = plt.subplots(
    2, 1, figsize=(13, 10), sharex=True,
    gridspec_kw={"hspace": 0.10},
)

for model in MODELS:
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    lc_pw = [LLAMACPP_THERMAL[m].get(model, (None, None))[0] for m in MODES]
    ol_pw = [OLLAMA_THERMAL[m].get(model, (None, None))[0]   for m in MODES]
    lc_tj = [LLAMACPP_THERMAL[m].get(model, (None, None))[1] for m in MODES]
    ol_tj = [OLLAMA_THERMAL[m].get(model, (None, None))[1]   for m in MODES]

    def vxy(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        return (list(a) for a in zip(*pairs)) if pairs else ([], [])

    xi, yi = vxy(X_B, lc_pw)
    if xi: ax_pw.plot(xi, yi, color=color, linestyle="-",  linewidth=2.5,
                      marker="o", markersize=8)
    xi, yi = vxy(X_B, ol_pw)
    if xi: ax_pw.plot(xi, yi, color=color, linestyle="--", linewidth=2.0,
                      marker="s", markersize=7, alpha=0.85)

    xi, yi = vxy(X_B, lc_tj)
    if xi: ax_tj.plot(xi, yi, color=color, linestyle="-",  linewidth=2.5,
                      marker="o", markersize=8)
    xi, yi = vxy(X_B, ol_tj)
    if xi: ax_tj.plot(xi, yi, color=color, linestyle="--", linewidth=2.0,
                      marker="s", markersize=7, alpha=0.85)

# TJ panel: 95 °C throttle line
ax_tj.axhline(95, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
ax_tj.text(0.01, 95.6, "95 °C throttle limit", transform=ax_tj.get_yaxis_transform(),
           color="red", fontsize=9, va="bottom")
ax_tj.set_ylim(40, 100)

# axis labels
ax_pw.set_ylabel("Avg VDD_CPU_GPU_CV Power (W)", fontsize=13)
ax_tj.set_ylabel("Peak Junction Temp TJ (°C)",   fontsize=13)
ax_tj.set_xlabel("Power Mode",                    fontsize=13)
ax_tj.set_xticks(X_B)
ax_tj.set_xticklabels(MXLAB, fontsize=13)
ax_pw.tick_params(axis="y", labelsize=11)
ax_tj.tick_params(axis="y", labelsize=11)

ax_pw.set_title(
    "Thermal profile — Avg Power (W) and Peak TJ (°C) across power modes\n"
    "Solid = llama.cpp  ·  Dashed = Ollama  ·  All 8 models",
    fontsize=13, pad=10,
)

# shared legend: same structure as Appendix A charts
model_handles = [
    mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=2.5,
                  marker="o", markersize=6, label=MODEL_LABELS[m])
    for m in MODELS
]
spacer = mlines.Line2D([], [], color="none", label="")
style_handles = [
    mlines.Line2D([], [], color="gray", linestyle="-",  linewidth=2.5,
                  marker="o", markersize=6, label="llama.cpp"),
    mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2.0,
                  marker="s", markersize=6, alpha=0.85, label="Ollama"),
]
ax_pw.legend(
    handles=model_handles + [spacer] + style_handles,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    fontsize=10,
    frameon=True,
    ncol=1,
    title="Model  /  Backend",
    title_fontsize=10,
    handlelength=2.2,
    handletextpad=0.7,
)

fig.tight_layout()
save(fig, "appendix_b_thermal_lines.png")

print("Done.")
