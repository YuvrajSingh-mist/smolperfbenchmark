#!/usr/bin/env python3
"""Standalone script to regenerate Figure 16 and Figure 17 for the
Jetson Orin Nano Super non-reasoning benchmark report.

Figure 16: Total energy per request vs output length at 25W, ctx=2048 (line chart)
Figure 17: Decode energy per output token in mJ (ctx=2048, gen=256) (bar chart)

Uses hardcoded data from generate_merged_charts.py (LLAMACPP_TOKJ, OLLAMA_TOKJ,
LLAMACPP_POWER, OLLAMA_POWER) plus derived/approximated total_j values for Figure 16.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

ROOT      = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
OUT_DIR   = ARTIFACTS / "charts"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    "smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "lfm2.5-350m",
    "lfm2.5-1.2b",  "qwen3-0.6b",   "llama3.2-1b",  "gemma3-1b",
]

MODEL_DISPLAY = {
    "smollm2-135m": "SmolLM2\n135M",
    "smollm2-360m": "SmolLM2\n360M",
    "qwen2.5-0.5b": "Qwen2.5\n0.5B",
    "lfm2.5-350m":  "LFM2.5\n350M",
    "lfm2.5-1.2b":  "LFM2.5\n1.2B",
    "qwen3-0.6b":   "Qwen3\n0.6B",
    "llama3.2-1b":  "Llama3.2\n1B",
    "gemma3-1b":    "Gemma3\n1B",
}
MDL = {k: v.replace("\n", " ") for k, v in MODEL_DISPLAY.items()}

GEN_LENGTHS = [64, 128, 256]
CANONICAL_CTX = 2048
CANONICAL_GEN = 256

MODEL_PAL = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))

MODE_STYLE = {
    "7W":          (":",  "^"),
    "15W":         ("-",  "o"),
    "25W":         ("--", "s"),
    "MAXN":        ("-.", "D"),
    "7W-ollama":   ("--", "v"),
    "15W-ollama":  ("--", "P"),
    "25W-ollama":  ("--", "h"),
    "MAXN-ollama": ("--", "X"),
}
MODE_PAL = {
    "7W":          "#2196F3",
    "15W":         "#4CAF50",
    "25W":         "#FF9800",
    "MAXN":        "#F44336",
    "7W-ollama":   "#7986CB",
    "15W-ollama":  "#009688",
    "25W-ollama":  "#FB8C00",
    "MAXN-ollama": "#E91E63",
}
BACKEND_ALPHA = {
    "7W": 1.0, "15W": 1.0, "25W": 1.0, "MAXN": 1.0,
    "7W-ollama": 0.65, "15W-ollama": 0.65, "25W-ollama": 0.65, "MAXN-ollama": 0.65,
}

sns.set_theme(style="darkgrid", font_scale=1.05)

# ── Hardcoded data from generate_merged_charts.py ────────────────────────────

LLAMACPP_TOKJ = {
    "7W":   {"smollm2-135m":21.7,"smollm2-360m":11.0,"qwen2.5-0.5b":7.3,"lfm2.5-350m":11.8,"lfm2.5-1.2b":4.8,"qwen3-0.6b":6.2,"llama3.2-1b":4.5,"gemma3-1b":4.9},
    "15W":  {"smollm2-135m":27.58,"smollm2-360m":14.71,"qwen2.5-0.5b":13.14,"lfm2.5-350m":16.19,"lfm2.5-1.2b":6.18,"qwen3-0.6b":6.83,"llama3.2-1b":5.34,"gemma3-1b":5.66},
    "25W":  {"smollm2-135m":29.62,"smollm2-360m":15.50,"qwen2.5-0.5b":13.30,"lfm2.5-350m":17.16,"lfm2.5-1.2b":6.37,"qwen3-0.6b":7.26,"llama3.2-1b":5.48,"gemma3-1b":6.02},
    "MAXN": {"smollm2-135m":24.72,"smollm2-360m":12.64,"qwen2.5-0.5b":11.78,"lfm2.5-350m":14.53,"lfm2.5-1.2b":5.37,"qwen3-0.6b":6.68,"llama3.2-1b":4.89,"gemma3-1b":5.19},
}

OLLAMA_TOKJ = {
    "7W":   {"smollm2-135m":19.21,"smollm2-360m":10.01,"qwen2.5-0.5b":7.80,"lfm2.5-350m":9.02,"lfm2.5-1.2b":4.77,"qwen3-0.6b":4.10,"llama3.2-1b":5.06,"gemma3-1b":4.76},
    "15W":  {"smollm2-135m":20.14,"smollm2-360m":10.41,"qwen2.5-0.5b":8.70,"lfm2.5-350m":6.64,"lfm2.5-1.2b":4.10,"qwen3-0.6b":6.78,"llama3.2-1b":5.38,"gemma3-1b":5.10},
    "25W":  {"smollm2-135m":21.26,"smollm2-360m":10.61,"qwen2.5-0.5b":8.95,"lfm2.5-350m":6.39,"lfm2.5-1.2b":3.94,"qwen3-0.6b":6.99,"llama3.2-1b":5.38,"gemma3-1b":5.29},
    "MAXN": {"smollm2-135m":18.65,"smollm2-360m":9.46,"qwen2.5-0.5b":7.84,"lfm2.5-350m":5.49,"lfm2.5-1.2b":3.43,"qwen3-0.6b":6.09,"llama3.2-1b":4.73,"gemma3-1b":4.46},
}

# ── Hardcoded total_j values for Figure 16 ────────────────────────────────────
# llama.cpp: from raw benchmark data (decode_power_W × OSL / tok_s + prefill_power_W × TTFT_s)
# Ollama: derived as llama_total_j × (llama_tok_s ÷ ollama_tok_s) — decode-dominated
# gen → {model: total_joules}

TOTAL_J_25W_LLAMACPP = {
    64: {
        "smollm2-135m": 7.8,
        "smollm2-360m": 12.3,
        "qwen2.5-0.5b": 14.2,
        "lfm2.5-350m": 11.9,
        "lfm2.5-1.2b": 18.4,
        "qwen3-0.6b": 15.7,
        "llama3.2-1b": 19.6,
        "gemma3-1b": 17.2,
    },
    128: {
        "smollm2-135m": 14.2,
        "smollm2-360m": 19.5,
        "qwen2.5-0.5b": 23.3,
        "lfm2.5-350m": 19.2,
        "lfm2.5-1.2b": 31.5,
        "qwen3-0.6b": 26.3,
        "llama3.2-1b": 32.7,
        "gemma3-1b": 28.7,
    },
    256: {
        "smollm2-135m": 25.1,
        "smollm2-360m": 35.7,
        "qwen2.5-0.5b": 42.4,
        "lfm2.5-350m": 34.5,
        "lfm2.5-1.2b": 55.9,
        "qwen3-0.6b": 49.5,
        "llama3.2-1b": 56.8,
        "gemma3-1b": 51.3,
    },
}

# 25W tok_s ratios (llamacpp ÷ ollama) from ratio_tables.md canonical cell
# Used to derive Ollama total_j: ollama_j ≈ llama_j × ratio (decode-dominated)
_TOKS_RATIO_25W = {
    "smollm2-135m": 1.37, "smollm2-360m": 1.46, "qwen2.5-0.5b": 1.67,
    "lfm2.5-350m": 4.19,  "lfm2.5-1.2b": 2.48,  "qwen3-0.6b": 1.06,
    "llama3.2-1b": 1.05,  "gemma3-1b": 1.17,
}

TOTAL_J_25W_OLLAMA = {}
for gen in GEN_LENGTHS:
    TOTAL_J_25W_OLLAMA[gen] = {
        m: round(TOTAL_J_25W_LLAMACPP[gen][m] * _TOKS_RATIO_25W[m], 1)
        for m in MODELS
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 16: Total energy per request vs output length at 25W, ctx=2048
# Solid lines = llama.cpp, dashed lines = Ollama, same color per model
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 16: Total energy vs gen length (25W, ctx=2048)")

fig, ax = plt.subplots(figsize=(13, 7))

for model in MODELS:
    color = MODEL_PAL[model]

    # llama.cpp — solid line, filled marker
    lc_x, lc_y = [], []
    for gen in GEN_LENGTHS:
        if model in TOTAL_J_25W_LLAMACPP[gen]:
            lc_x.append(gen)
            lc_y.append(TOTAL_J_25W_LLAMACPP[gen][model])
    if lc_x:
        ax.plot(lc_x, lc_y, marker="o", ls="-", lw=2.5,
                color=color, label=f"{MDL[model]} (llama.cpp)", ms=9,
                markerfacecolor="white", markeredgewidth=2)

    # Ollama — dashed line, open marker, lower alpha
    ol_x, ol_y = [], []
    for gen in GEN_LENGTHS:
        if model in TOTAL_J_25W_OLLAMA[gen]:
            ol_x.append(gen)
            ol_y.append(TOTAL_J_25W_OLLAMA[gen][model])
    if ol_x:
        ax.plot(ol_x, ol_y, marker="s", ls="--", lw=2.0,
                color=color, label=f"{MDL[model]} (Ollama)", ms=8,
                alpha=0.65)

ax.set_title("Total Energy per Request vs Gen Length\n"
             "25W power mode, ctx=2048 tok — solid=llama.cpp, dashed=Ollama",
             fontweight="bold", fontsize=14, pad=12)
ax.set_xlabel("Generation length (output tokens)", fontsize=12)
ax.set_ylabel("Total energy per request (J)", fontsize=12)
ax.set_xticks(GEN_LENGTHS)
ax.set_xticklabels([str(g) for g in GEN_LENGTHS], fontsize=11)
ax.tick_params(axis="y", labelsize=10)
ax.set_xlim(48, 272)

# Legend: llama.cpp = solid ○, Ollama = dashed □
lc_handle = mlines.Line2D([], [], color="gray", ls="-", lw=2.5, marker="o",
                          ms=8, markerfacecolor="white", markeredgewidth=2,
                          label="llama.cpp (solid)")
ol_handle = mlines.Line2D([], [], color="gray", ls="--", lw=2.0, marker="s",
                          ms=7, alpha=0.65, label="Ollama (dashed)")
spacer = mlines.Line2D([], [], color="none", label="")
model_handles = [mlines.Line2D([], [], color=MODEL_PAL[m], ls="-", lw=3, label=MDL[m])
                 for m in MODELS]

ax.legend(handles=model_handles + [spacer, lc_handle, ol_handle],
          loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=10, frameon=True, title="Model / Backend", title_fontsize=11,
          borderaxespad=0)

fig.tight_layout()
path = OUT_DIR / "E_total_energy_vs_gen_length.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved {path.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 17: Decode energy per output token in mJ (ctx=2048, gen=256)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 17: mJ per output token (ctx=2048, gen=256)")

# Build list of all modes (llamacpp + ollama)
LC_MODES = ["7W", "15W", "25W", "MAXN"]
OL_MODES = ["7W-ollama", "15W-ollama", "25W-ollama", "MAXN-ollama"]
ALL_MODES = LC_MODES + OL_MODES

# mJ per output token = 1000 / tok_j
def mj_from_tokj(tokj_dict, mode, model):
    v = tokj_dict.get(mode, {}).get(model)
    if v and v > 0:
        return 1000.0 / v
    return np.nan

n_models = len(MODELS)
n_modes = len(ALL_MODES)
WIDTH = 0.10
x = np.arange(n_models)
offsets = np.linspace(-(n_modes - 1) * WIDTH / 2, (n_modes - 1) * WIDTH / 2, n_modes)

fig, ax = plt.subplots(figsize=(22, 7))

for i, mode in enumerate(ALL_MODES):
    is_ollama = mode.endswith("-ollama")
    base_mode = mode.replace("-ollama", "")
    tokj_dict = OLLAMA_TOKJ if is_ollama else LLAMACPP_TOKJ

    vals = []
    for m in MODELS:
        vals.append(mj_from_tokj(tokj_dict, base_mode, m))

    al = BACKEND_ALPHA.get(mode, 1.0)
    ls, mk = MODE_STYLE.get(mode, ("-", "o"))
    # Ollama uses hatch pattern to distinguish from solid llamacpp
    hatch = "//" if is_ollama else ""

    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8,
                  alpha=al, hatch=hatch)

    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.015,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", rotation=90, alpha=min(al + 0.1, 1.0))

ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=10)
ax.set_title("Decode Energy per Output Token\n"
             f"All Power Modes — ctx={CANONICAL_CTX}, gen={CANONICAL_GEN} — "
             "solid=llama.cpp, hatched=Ollama",
             fontweight="bold", fontsize=13, pad=10)
ax.set_ylabel("mJ per output token (lower = better)", fontsize=12)

# Legend: modes with backend distinction
legend_handles = []
for mode in ALL_MODES:
    is_ollama = mode.endswith("-ollama")
    ls, mk = MODE_STYLE.get(mode, ("-", "o"))
    al = BACKEND_ALPHA.get(mode, 1.0)
    hatch = "//" if is_ollama else ""
    legend_handles.append(
        mlines.Line2D([], [], color=MODE_PAL[mode], marker=mk, ls=ls, lw=2,
                      label=mode, ms=7, alpha=al,
                      markerfacecolor=MODE_PAL[mode] if not is_ollama else "white")
    )

ax.legend(handles=legend_handles, fontsize=9, loc="upper left",
          bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
          title="Power Mode / Backend", title_fontsize=10,
          ncol=1)

fig.tight_layout()
path = OUT_DIR / "E_mj_per_output_token.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved {path.name}")

print("\nDone. Figure 16 and Figure 17 regenerated.")
