#!/usr/bin/env python3
"""Generate merged dual-y-axis charts.

Charts produced:
  appendix_a_toks_ttft.png   — tok/s (top) + TTFT ms (bottom), x=modes  [A.1+A.2]
  appendix_a_power_tokj.png  — Power W (top) + tok/J (bottom),  x=modes  [A.4+A.5]
  appendix_c_ratio_chart.png — tok/s ratio llama.cpp ÷ Ollama (top) + tok/J ratio llama.cpp ÷ Ollama (bottom), x=models [C]
  appendix_h_prefill_ttft.png — Prefill tok/s (top) + TTFT ms (bottom), x=modes  [H]

Anti-spaghetti design rules:
  - tok/s / Prefill go UP with mode; TTFT goes DOWN → natural X-pattern: the two
    groups sit in opposite corners at 7W and MAXN so they never fully overlap.
  - Power+tok/J: tok/J peaks at 25W then drops while Power keeps climbing → visual
    divergence at MAXN tells the story. Alpha 1.0 (Power, thick) vs 0.5 (tok/J, thin).
  - C ratios: only 4 lines per metric (one per mode) → inherently uncluttered.
  - figsize=(13,7) throughout; consistent legend style across all charts.
"""

import json, re, statistics
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

ROOT      = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
OUT_DIR   = ARTIFACTS / "charts"
OUT_DIR.mkdir(exist_ok=True)

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
    "lfm2.5-1.2b",  "qwen3-0.6b",   "llama3.2-1b",  "gemma3-1b",
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
MODES    = ["7W", "15W", "25W", "MAXN"]
MXLAB    = ["7W", "15W", "25W", "MAXN"]
X        = np.arange(len(MODES))

MODEL_COLORS = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
MODE_COLORS  = dict(zip(MODES, sns.color_palette("tab10", 4)))

sns.set_theme(style="darkgrid", font_scale=1.05)


# ── hardcoded power and tok/J (verified against raw data) ─────────────────────
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


# ── data loader ────────────────────────────────────────────────────────────────

def load_p50(art_dir, model, subdir=""):
    """Return (tok_s, ttft_ms, itl_ms, prefill_toks) p50 from aiperf JSON."""
    base = ARTIFACTS / art_dir
    p = (base / subdir / model / "gen256" / "ctx2048" / "profile_export_aiperf.json"
         if subdir else
         base / model / "gen256" / "ctx2048" / "profile_export_aiperf.json")
    if not p.exists():
        return None, None, None, None
    d = json.loads(p.read_text())
    def g(k): return (d.get(k) or {}).get("p50")
    return (g("output_token_throughput_per_user"),
            g("time_to_first_token"),
            g("inter_token_latency"),
            g("prefill_throughput_per_user"))


print("Loading data...")
lc, ol = {}, {}
for src, dirs, sub in [("lc", LLAMACPP_DIRS, ""), ("ol", OLLAMA_DIRS, "ollama")]:
    store = lc if src == "lc" else ol
    for metric in ("toks", "ttft", "itl", "prefill"):
        store[metric] = {}
    for mode in MODES:
        for d in ("toks", "ttft", "itl", "prefill"):
            store[d][mode] = {}
        art = dirs[mode]
        for model in MODELS:
            ts, tt, it, pf = load_p50(art, model, sub)
            store["toks"][mode][model]   = ts
            store["ttft"][mode][model]   = tt
            store["itl"][mode][model]    = it
            store["prefill"][mode][mode] = pf   # keyed differently below
        for model in MODELS:
            _, _, _, pf = load_p50(art, model, sub)
            store["prefill"][mode][model] = pf


# ── shared helpers ─────────────────────────────────────────────────────────────

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def vxy(xs, ys):
    """Return (x_list, y_list) with None values removed."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pairs:
        return [], []
    xx, yy = zip(*pairs)
    return list(xx), list(yy)


def side_legend(ax, model_handles, style_handles, title="Model  /  Style"):
    """Attach two-section legend to right of axes."""
    spacer = mlines.Line2D([], [], color="none", label="")
    ax.legend(
        handles=model_handles + [spacer] + style_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=10,
        frameon=True,
        ncol=1,
        title=title,
        title_fontsize=10,
        handlelength=2.2,
        handletextpad=0.7,
    )


def model_legend_handles():
    return [
        mlines.Line2D([], [], color=MODEL_COLORS[m], linewidth=2.5,
                      marker="o", markersize=6, label=MODEL_LABELS[m])
        for m in MODELS
    ]


# ── Chart 1: tok/s (top) + TTFT ms (bottom)  [Appendix A.1] ─────────────────
# Two stacked panels, shared x=modes. 16 lines per panel (8 models × 2 backends).
# Same structure as Appendix B which came out clean.

print("Generating appendix_a_toks_ttft.png ...")

fig, (ax_ts, ax_tt) = plt.subplots(
    2, 1, figsize=(13, 10), sharex=True,
    gridspec_kw={"hspace": 0.10},
)

for model in MODELS:
    color = MODEL_COLORS[model]
    lc_ts = [lc["toks"][m][model] for m in MODES]
    ol_ts = [ol["toks"][m][model] for m in MODES]
    lc_tt = [lc["ttft"][m][model] for m in MODES]
    ol_tt = [ol["ttft"][m][model] for m in MODES]

    xi, yi = vxy(X, lc_ts); ax_ts.plot(xi, yi, color=color, ls="-",  lw=2.5, marker="o", ms=8)
    xi, yi = vxy(X, ol_ts); ax_ts.plot(xi, yi, color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.85)
    xi, yi = vxy(X, lc_tt); ax_tt.plot(xi, yi, color=color, ls="-",  lw=2.5, marker="o", ms=8)
    xi, yi = vxy(X, ol_tt); ax_tt.plot(xi, yi, color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.85)

ax_ts.set_ylabel("Output Tok/s", fontsize=13)
ax_tt.set_ylabel("TTFT p50 (ms)", fontsize=13)
ax_tt.set_xlabel("Power Mode", fontsize=13)
ax_tt.set_xticks(X); ax_tt.set_xticklabels(MXLAB, fontsize=13)
ax_ts.tick_params(axis="y", labelsize=11)
ax_tt.tick_params(axis="y", labelsize=11)
ax_ts.set_title(
    "Output Tok/s (top) and TTFT p50 ms (bottom) — llama.cpp vs Ollama\n"
    "Solid = llama.cpp  ·  Dashed = Ollama  ·  ctx=2048 gen=256",
    fontsize=13, pad=10,
)

spacer = mlines.Line2D([], [], color="none", label="")
style_handles = [
    mlines.Line2D([], [], color="gray", ls="-",  lw=2.5, marker="o", ms=6, label="llama.cpp"),
    mlines.Line2D([], [], color="gray", ls="--", lw=2.0, marker="s", ms=6, alpha=0.85, label="Ollama"),
]
ax_ts.legend(
    handles=model_legend_handles() + [spacer] + style_handles,
    loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    fontsize=10, frameon=True, ncol=1,
    title="Model  /  Backend", title_fontsize=10,
    handlelength=2.2, handletextpad=0.7,
)
fig.tight_layout()
save(fig, "appendix_a_toks_ttft.png")


# ── Chart 2: Power W (top) + tok/J (bottom)  [Appendix A.3] ─────────────────
# Two stacked panels. Top: decode power; Bottom: tok/J.
# The key story (tok/J peaks at 25W then drops while Power keeps climbing) is
# immediately readable when the two metrics are on separate clean panels.

print("Generating appendix_a_power_tokj.png ...")

fig, (ax_pw, ax_tj) = plt.subplots(
    2, 1, figsize=(13, 10), sharex=True,
    gridspec_kw={"hspace": 0.10},
)

for model in MODELS:
    color = MODEL_COLORS[model]
    lc_pw = [LLAMACPP_POWER[m].get(model) for m in MODES]
    ol_pw = [OLLAMA_POWER[m].get(model)   for m in MODES]
    lc_tj = [LLAMACPP_TOKJ[m].get(model)  for m in MODES]
    ol_tj = [OLLAMA_TOKJ[m].get(model)    for m in MODES]

    xi, yi = vxy(X, lc_pw); ax_pw.plot(xi, yi, color=color, ls="-",  lw=2.5, marker="o", ms=8)
    xi, yi = vxy(X, ol_pw); ax_pw.plot(xi, yi, color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.85)
    xi, yi = vxy(X, lc_tj); ax_tj.plot(xi, yi, color=color, ls="-",  lw=2.5, marker="o", ms=8)
    xi, yi = vxy(X, ol_tj); ax_tj.plot(xi, yi, color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.85)

ax_pw.set_ylabel("Decode Power (W)", fontsize=13)
ax_tj.set_ylabel("Output Tok/J",     fontsize=13)
ax_tj.set_xlabel("Power Mode", fontsize=13)
ax_tj.set_xticks(X); ax_tj.set_xticklabels(MXLAB, fontsize=13)
ax_pw.tick_params(axis="y", labelsize=11)
ax_tj.tick_params(axis="y", labelsize=11)
ax_pw.set_title(
    "Decode Power W (top) and Output Tok/J (bottom) — llama.cpp vs Ollama\n"
    "tok/J peaks at 25W then drops at MAXN while Power keeps climbing  ·  Solid = llama.cpp  ·  Dashed = Ollama",
    fontsize=13, pad=10,
)

spacer = mlines.Line2D([], [], color="none", label="")
style_handles = [
    mlines.Line2D([], [], color="gray", ls="-",  lw=2.5, marker="o", ms=6, label="llama.cpp"),
    mlines.Line2D([], [], color="gray", ls="--", lw=2.0, marker="s", ms=6, alpha=0.85, label="Ollama"),
]
ax_pw.legend(
    handles=model_legend_handles() + [spacer] + style_handles,
    loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    fontsize=10, frameon=True, ncol=1,
    title="Model  /  Backend", title_fontsize=10,
    handlelength=2.2, handletextpad=0.7,
)
fig.tight_layout()
save(fig, "appendix_a_power_tokj.png")


# ── Chart 3: tok/s ratio llama.cpp ÷ Ollama (top) + tok/J ratio llama.cpp ÷ Ollama (bottom)  [Appendix C] ───────
# Two stacked panels, shared x=modes. One line per model (consistent with A.1, A.3, H.0).
# Reference line at 1.0 on both panels. X = 4 power modes.

print("Generating appendix_c_ratio_chart.png ...")

fig, (ax_ts, ax_tj) = plt.subplots(
    2, 1, figsize=(13, 10), sharex=True,
    gridspec_kw={"hspace": 0.10},
)

for model in MODELS:
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    ts_ratios = []
    tj_ratios = []
    for mode in MODES:
        lc_v = lc["toks"][mode].get(model)
        ol_v = ol["toks"][mode].get(model)
        ts_ratios.append(lc_v / ol_v if (lc_v and ol_v) else None)

        lc_j = LLAMACPP_TOKJ[mode].get(model)
        ol_j = OLLAMA_TOKJ[mode].get(model)
        tj_ratios.append(lc_j / ol_j if (lc_j and ol_j) else None)

    xi, yi = vxy(X, ts_ratios)
    if xi:
        ax_ts.plot(xi, yi, color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)

    xi, yi = vxy(X, tj_ratios)
    if xi:
        ax_tj.plot(xi, yi, color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)

# 1× reference on both panels
ax_ts.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.5)
ax_tj.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.5)

ax_ts.set_ylabel("Tok/s ratio  llama.cpp ÷ Ollama", fontsize=13)
ax_tj.set_ylabel("Tok/J ratio  llama.cpp ÷ Ollama", fontsize=13)
ax_tj.set_xlabel("Power Mode", fontsize=13)
ax_tj.set_xticks(X)
ax_tj.set_xticklabels(MXLAB, fontsize=13)
ax_ts.tick_params(axis="y", labelsize=11)
ax_tj.tick_params(axis="y", labelsize=11)
ax_ts.set_title(
    "llama.cpp ÷ Ollama ratios — Tok/s (top) and Tok/J (bottom)\n"
    "> 1× means llama.cpp leads  ·  dotted line = parity  ·  ctx=2048 gen=256",
    fontsize=13, pad=10,
)

model_handles = [
    mlines.Line2D([], [], color=MODEL_COLORS[m], lw=2.5, marker="o", ms=6, label=MODEL_LABELS[m])
    for m in MODELS
]
ax_ts.legend(
    handles=model_handles,
    loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    fontsize=10, frameon=True, ncol=1,
    title="Model", title_fontsize=10,
    handlelength=2.2, handletextpad=0.7,
)
fig.tight_layout()
save(fig, "appendix_c_ratio_chart.png")


# ── Chart 3b: Prefill throughput ratio llama.cpp ÷ Ollama  [Appendix C.1] ─────
# Single panel, x=modes, one line per model. Reference line at 1.0.
# Ratio = llama.cpp ÷ Ollama; > 1× means llama.cpp prefill is faster.

print("Generating appendix_c1_prefill_ratio_chart.png ...")

fig, ax = plt.subplots(1, 1, figsize=(13, 6))

for model in MODELS:
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    pf_ratios = []
    for mode in MODES:
        lc_v = lc["prefill"][mode].get(model)
        ol_v = ol["prefill"][mode].get(model)
        pf_ratios.append(lc_v / ol_v if (lc_v and ol_v) else None)

    xi, yi = vxy(X, pf_ratios)
    if xi:
        ax.plot(xi, yi, color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)

ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.5)
ax.set_ylabel("Prefill throughput ratio  llama.cpp ÷ Ollama", fontsize=13)
ax.set_xlabel("Power Mode", fontsize=13)
ax.set_xticks(X)
ax.set_xticklabels(MXLAB, fontsize=13)
ax.tick_params(axis="y", labelsize=11)
ax.set_title(
    "llama.cpp ÷ Ollama — Prefill throughput ratio\n"
    "> 1× means llama.cpp prefill is faster  ·  dotted line = parity  ·  ctx=2048 gen=256",
    fontsize=13, pad=10,
)

model_handles = [
    mlines.Line2D([], [], color=MODEL_COLORS[m], lw=2.5, marker="o", ms=6, label=MODEL_LABELS[m])
    for m in MODELS
]
ax.legend(
    handles=model_handles,
    loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    fontsize=10, frameon=True, ncol=1,
    title="Model", title_fontsize=10,
    handlelength=2.2, handletextpad=0.7,
)
fig.tight_layout()
save(fig, "appendix_c1_prefill_ratio_chart.png")


# ── Chart 4: Prefill tok/s (top) + TTFT ms (bottom)  [Appendix H] ──────────
# Two stacked panels, shared x=modes. Same clean structure as A.1 and B.

print("Generating appendix_h_prefill_ttft.png ...")

fig, (ax_pf, ax_tt) = plt.subplots(
    2, 1, figsize=(13, 10), sharex=True,
    gridspec_kw={"hspace": 0.10},
)

for model in MODELS:
    color = MODEL_COLORS[model]
    lc_pf = [lc["prefill"][m][model] for m in MODES]
    ol_pf = [ol["prefill"][m][model] for m in MODES]
    lc_tt = [lc["ttft"][m][model]    for m in MODES]
    ol_tt = [ol["ttft"][m][model]    for m in MODES]

    xi, yi = vxy(X, lc_pf); ax_pf.plot(xi, yi, color=color, ls="-",  lw=2.5, marker="o", ms=8)
    xi, yi = vxy(X, ol_pf); ax_pf.plot(xi, yi, color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.85)
    xi, yi = vxy(X, lc_tt); ax_tt.plot(xi, yi, color=color, ls="-",  lw=2.5, marker="o", ms=8)
    xi, yi = vxy(X, ol_tt); ax_tt.plot(xi, yi, color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.85)

ax_pf.set_ylabel("Prefill Throughput (tok/s)", fontsize=13)
ax_tt.set_ylabel("TTFT p50 (ms)",              fontsize=13)
ax_tt.set_xlabel("Power Mode", fontsize=13)
ax_tt.set_xticks(X); ax_tt.set_xticklabels(MXLAB, fontsize=13)
ax_pf.tick_params(axis="y", labelsize=11)
ax_tt.tick_params(axis="y", labelsize=11)
ax_pf.set_title(
    "Prefill Throughput tok/s (top) and TTFT p50 ms (bottom) — llama.cpp vs Ollama\n"
    "Faster prefill directly lowers TTFT  ·  Solid = llama.cpp  ·  Dashed = Ollama  ·  ctx=2048 gen=256",
    fontsize=13, pad=10,
)

spacer = mlines.Line2D([], [], color="none", label="")
style_handles = [
    mlines.Line2D([], [], color="gray", ls="-",  lw=2.5, marker="o", ms=6, label="llama.cpp"),
    mlines.Line2D([], [], color="gray", ls="--", lw=2.0, marker="s", ms=6, alpha=0.85, label="Ollama"),
]
ax_pf.legend(
    handles=model_legend_handles() + [spacer] + style_handles,
    loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    fontsize=10, frameon=True, ncol=1,
    title="Model  /  Backend", title_fontsize=10,
    handlelength=2.2, handletextpad=0.7,
)
fig.tight_layout()
save(fig, "appendix_h_prefill_ttft.png")


# ── Chart 5: TTFT ratio + ITL ratio + Power ratio  [Appendix C.3] ──────────
# Three stacked panels, shared x=modes. One line per model (consistent with C.0).
# TTFT and ITL: ratio = llama.cpp ÷ Ollama; < 1× = llama.cpp is faster.
# Power: ratio = llama.cpp ÷ Ollama; > 1× = llama.cpp draws more power.

print("Generating appendix_c3_latency_power_ratio.png ...")

fig, (ax_tt, ax_il, ax_pw) = plt.subplots(
    3, 1, figsize=(13, 13), sharex=True,
    gridspec_kw={"hspace": 0.12},
)

for model in MODELS:
    color = MODEL_COLORS[model]
    label = MODEL_LABELS[model]

    tt_ratios, il_ratios, pw_ratios = [], [], []
    for mode in MODES:
        lc_tt = lc["ttft"][mode].get(model)
        ol_tt = ol["ttft"][mode].get(model)
        tt_ratios.append(lc_tt / ol_tt if (lc_tt and ol_tt) else None)

        lc_il = lc["itl"][mode].get(model)
        ol_il = ol["itl"][mode].get(model)
        il_ratios.append(lc_il / ol_il if (lc_il and ol_il) else None)

        lc_pw = LLAMACPP_POWER[mode].get(model)
        ol_pw = OLLAMA_POWER[mode].get(model)
        pw_ratios.append(lc_pw / ol_pw if (lc_pw and ol_pw) else None)

    xi, yi = vxy(X, tt_ratios)
    if xi:
        ax_tt.plot(xi, yi, color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)
    xi, yi = vxy(X, il_ratios)
    if xi:
        ax_il.plot(xi, yi, color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)
    xi, yi = vxy(X, pw_ratios)
    if xi:
        ax_pw.plot(xi, yi, color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)

for ax in (ax_tt, ax_il, ax_pw):
    ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.5)

ax_tt.set_ylabel("'TTFT' ratio  llama.cpp ÷ Ollama", fontsize=13)
ax_il.set_ylabel("'ITL' ratio  llama.cpp ÷ Ollama", fontsize=13)
ax_pw.set_ylabel("'Power' ratio  llama.cpp ÷ Ollama", fontsize=13)
ax_pw.set_xlabel("Power Mode", fontsize=13)
ax_pw.set_xticks(X)
ax_pw.set_xticklabels(MXLAB, fontsize=13)
for ax in (ax_tt, ax_il, ax_pw):
    ax.tick_params(axis="y", labelsize=11)
ax_tt.set_title(
    "llama.cpp ÷ Ollama ratios — 'TTFT' (top), 'ITL' (middle), 'Power' (bottom)\n"
    "< 1× = llama.cpp faster/lower  ·  dotted line = parity  ·  ctx=2048 gen=256",
    fontsize=13, pad=10,
)
model_handles = [
    mlines.Line2D([], [], color=MODEL_COLORS[m], lw=2.5, marker="o", ms=6, label=MODEL_LABELS[m])
    for m in MODELS
]
ax_tt.legend(
    handles=model_handles,
    loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    fontsize=10, frameon=True, ncol=1,
    title="Model", title_fontsize=10,
    handlelength=2.2, handletextpad=0.7,
)
fig.tight_layout()
save(fig, "appendix_c3_latency_power_ratio.png")

print("Done.")
