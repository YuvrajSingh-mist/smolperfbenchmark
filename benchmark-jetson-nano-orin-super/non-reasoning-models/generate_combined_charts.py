#!/usr/bin/env python3
"""Generate 4-power-mode comparison charts for the Jetson Orin Nano Super benchmark.

Canonical standard: ctx=2048, gen=256 (highest sweep point).
Also generates all 12 prompt×gen combination charts for Appendix E.
"""

import json, re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT      = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
OUT_DIR   = ARTIFACTS / "charts"
OUT_DIR.mkdir(exist_ok=True)

RUNS = {
    "7W":   "blog-all-20260527-0447-7w",
    "15W":  "blog-all-20260525-2325-15w",
    "25W":  "blog-all-20260527-0139-25w",
    "MAXN": "blog-all-20260526-0445-maxn",
}

MODELS = [
    "smollm2-135m",
    "smollm2-360m",
    "qwen2.5-0.5b",
    "lfm2.5-350m",
    "lfm2.5-1.2b",
    "qwen3-0.6b",
    "llama3.2-1b",
    "gemma3-1b",
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

PROMPT_LENGTHS = [128, 512, 1024, 2048]
GEN_LENGTHS    = [64, 128, 256]

CANONICAL_CTX = 2048
CANONICAL_GEN = 256

MODE_STYLE = {
    "7W":   (":",  "^"),
    "15W":  ("-",  "o"),
    "25W":  ("--", "s"),
    "MAXN": ("-.", "D"),
}
MODE_PAL = {
    "7W":   "#2196F3",
    "15W":  "#4CAF50",
    "25W":  "#FF9800",
    "MAXN": "#F44336",
}
MODEL_PAL = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
WIDTH = 0.2

sns.set_theme(style="darkgrid", font_scale=1.05)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── load tegrastats per run ────────────────────────────────────────────────────
_tcache = {}
def get_tegra(art):
    if art not in _tcache:
        records = []
        p = ARTIFACTS / art / "tegrastats.log"
        if p.exists():
            with open(p) as f:
                for line in f:
                    m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
                    if not m:
                        continue
                    ep = datetime.strptime(m.group(1), '%m-%d-%Y %H:%M:%S').timestamp()
                    pw = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
                    tj = re.search(r'tj@([\d.]+)C', line)
                    if pw:
                        records.append((ep, int(pw.group(1)), float(tj.group(1)) if tj else None))
        _tcache[art] = records
    return _tcache[art]


# ── build dataframe ────────────────────────────────────────────────────────────
rows = []
for mode, art in RUNS.items():
    tegra = get_tegra(art)
    for model in MODELS:
        for gen in GEN_LENGTHS:
            for ctx in PROMPT_LENGTHS:
                p = ARTIFACTS / art / model / f"gen{gen}" / f"ctx{ctx}" / "profile_export_aiperf.json"
                if not p.exists():
                    continue
                try:
                    d = json.loads(p.read_text())
                except Exception:
                    continue

                def pct(k, v="avg"):
                    return (d.get(k) or {}).get(v)

                tok_s   = pct("output_token_throughput_per_user")
                ttft    = pct("time_to_first_token",  "p50")   # p50 more robust than mean
                ttft_p90= pct("time_to_first_token",  "p90")
                ttft_p99= pct("time_to_first_token",  "p99")
                itl     = pct("inter_token_latency",  "p50")   # p50 more robust than mean
                itl_p99 = pct("inter_token_latency",  "p99")
                prefill = pct("prefill_throughput_per_user")
                rl_avg  = pct("request_latency", "p50")         # p50 for latency
                rl_mean = pct("request_latency", "avg")        # mean for energy calc
                e2e     = pct("e2e_output_token_throughput")
                isl     = pct("input_sequence_length")
                osl     = pct("output_sequence_length")

                t0_str = d.get("start_time")
                t1_str = d.get("end_time")
                if not t0_str or not t1_str:
                    continue
                t0 = datetime.fromisoformat(t0_str).timestamp()
                t1 = datetime.fromisoformat(t1_str).timestamp()

                samps = [(mw, tj) for (ep, mw, tj) in tegra if t0 <= ep <= t1]
                if not samps:
                    continue
                power_w = sum(mw for mw, _ in samps) / len(samps) / 1000
                tok_j   = tok_s / power_w if (tok_s and power_w) else None

                # Energy metrics (J)
                rl_s = (rl_avg or 0) / 1000.0
                ttft_s = (ttft or 0) / 1000.0
                prefill_j = power_w * ttft_s if (power_w and ttft_s) else None
                total_j   = power_w * rl_s if (power_w and rl_s) else None
                decode_j  = (total_j - prefill_j) if (total_j and prefill_j) else None

                # Phase tok/J
                prefill_tokj = isl / prefill_j if (isl and prefill_j and prefill_j > 0) else None
                decode_tokj  = osl / decode_j if (osl and decode_j and decode_j > 0) else None
                total_tokj   = (isl + osl) / total_j if (isl is not None and osl is not None and total_j and total_j > 0) else None

                rows.append(dict(
                    mode=mode, model=model,
                    prompt=ctx, gen=gen,
                    tok_s=tok_s, ttft=ttft,
                    ttft_p90=ttft_p90, ttft_p99=ttft_p99,
                    itl=itl, itl_p99=itl_p99,
                    prefill=prefill,
                    rl_avg=rl_avg, e2e=e2e,
                    power_w=power_w, tok_j=tok_j,
                    isl=isl, osl=osl,
                    prefill_j=prefill_j, decode_j=decode_j, total_j=total_j,
                    prefill_tokj=prefill_tokj, decode_tokj=decode_tokj, total_tokj=total_tokj,
                ))

df = pd.DataFrame(rows)
modes_avail = [m for m in ["7W", "15W", "25W", "MAXN"] if m in df["mode"].unique()]
n = len(modes_avail)
offsets = np.linspace(-(n - 1) * WIDTH / 2, (n - 1) * WIDTH / 2, n)
print(f"Loaded {len(df)} rows across {df['mode'].nunique()} modes, {df['model'].nunique()} models")


def mode_legend_handles():
    return [mlines.Line2D([], [], color=MODE_PAL[m], ls=ls, marker=mk, lw=2, label=m, ms=7)
            for m, (ls, mk) in MODE_STYLE.items() if m in modes_avail]

def model_legend_handles(models):
    return [mlines.Line2D([], [], color=MODEL_PAL[m], ls="-", lw=3,
                          label=MDL[m])
            for m in models if m in MODEL_PAL]


def canonical_bar(ctx_val, gen_val, fname_prefix, title_suffix=""):
    """Two-panel bar chart: output tok/s + output tok/J for a specific ctx, gen."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(len(MODELS))
    for i, mode in enumerate(modes_avail):
        toks_vals, tokj_vals = [], []
        for m in MODELS:
            sub = df[(df.model == m) & (df["mode"] == mode) &
                     (df.prompt == ctx_val) & (df.gen == gen_val)]
            toks_vals.append(sub["tok_s"].mean() if not sub.empty else np.nan)
            tokj_vals.append(sub["tok_j"].mean() if not sub.empty else np.nan)
        bars1 = ax1.bar(x + offsets[i], toks_vals, WIDTH, label=mode,
                        color=MODE_PAL[mode], edgecolor="white")
        bars2 = ax2.bar(x + offsets[i], tokj_vals, WIDTH, label=mode,
                        color=MODE_PAL[mode], edgecolor="white")
        for bar, v in zip(bars1, toks_vals):
            if not np.isnan(v):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
        for bar, v in zip(bars2, tokj_vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=9)
    ax1.set_title(f"Output Tok/s: ctx={ctx_val}, gen={gen_val}", fontweight="bold")
    ax1.set_ylabel("Output tokens per second")
    ax2.set_title(f"Output Tok/J: ctx={ctx_val}, gen={gen_val}", fontweight="bold")
    ax2.set_ylabel("Output tokens per joule")
    fig.suptitle(
        f"All 4 Power Modes: ctx={ctx_val} tok prompt, gen={gen_val} tok output{title_suffix}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, f"{fname_prefix}_ctx{ctx_val}_gen{gen_val}.png")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHARTS  (canonical: ctx=2048, gen=256)
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Main charts (canonical ctx=2048, gen=256) ──")

# ── Helper: faceted 2x4 line chart with fixed y-scale ─────────────────────────
def faceted_line_chart(df, y_field, y_label, gen_val, fname, y_scale=None, x_is_prompt=True):
    """2×4 facet line chart, sharey=True, fixed y-scale. x = prompt lengths if x_is_prompt else gen lengths."""
    x_vals = PROMPT_LENGTHS if x_is_prompt else GEN_LENGTHS
    x_label = "Prompt (tok)" if x_is_prompt else "Gen (tok)"
    data = df[(df.gen == gen_val) & (df["mode"].isin(modes_avail))][y_field].dropna()
    if data.empty:
        return
    y_max = y_scale if y_scale else data.max() * 1.15
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    fig.suptitle(f"{y_label} vs {'Prompt' if x_is_prompt else 'Gen'} Length: All Power Modes (gen={gen_val} tok)",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 4][idx % 4]
        sub = df[(df.model == model) & (df.gen == gen_val) & (df["mode"].isin(modes_avail))]
        if x_is_prompt:
            for mode, (ls, mk) in MODE_STYLE.items():
                s = sub[(sub["mode"] == mode)].sort_values("prompt")
                if s.empty: continue
                ax.plot(s.prompt, s[y_field], marker=mk, ls=ls, lw=2,
                        color=MODE_PAL[mode], label=mode, ms=7)
        else:
            for mode, (ls, mk) in MODE_STYLE.items():
                s = sub[(sub["mode"] == mode)].sort_values("gen")
                if s.empty: continue
                ax.plot(s.gen, s[y_field], marker=mk, ls=ls, lw=2,
                        color=MODE_PAL[mode], label=mode, ms=7)
        ax.set_title(MDL[model], fontsize=10, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label if idx % 4 == 0 else "", fontsize=8)
        ax.set_xticks(x_vals)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    save(fig, fname)

# 1. Output Tok/s vs Prompt: 2x4 facet, gen=256
faceted_line_chart(df, "tok_s", "Output Tok/s", 256, "1_tok_s_vs_prompt_gen256.png")

# 2. Output Tok/J vs Prompt: 2x4 facet, gen=256
faceted_line_chart(df, "tok_j", "Output Tok/J", 256, "2_tok_j_vs_prompt_gen256.png")

# 3. Best Tok/J grouped bar
best = df.groupby(["model", "mode"])["tok_j"].max().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = [best.loc[m, mode] if (m in best.index and mode in best.columns
                                   and not pd.isna(best.loc[m, mode])) else 0
            for m in MODELS]
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Best Output Tok/J per Model: All Power Modes",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Output Tok/J (tokens per joule)")
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "3_best_tok_j_bar.png")

# 4. Average power grouped bar
avg_pw = df.groupby(["model", "mode"])["power_w"].mean().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = [avg_pw.loc[m, mode] if (m in avg_pw.index and mode in avg_pw.columns
                                     and not pd.isna(avg_pw.loc[m, mode])) else 0
            for m in MODELS]
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{v:.1f}W", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Average Power Draw per Model: All Power Modes (VDD_CPU_GPU_CV)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Power (W)")
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "4_avg_power_bar.png")

# 5. TTFT p50 grouped bar — ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(15, 6))
x_bar = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["ttft"].mean() if not sub.empty else np.nan)
    bars = ax.bar(x_bar + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+bar.get_height()*0.01,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x_bar)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"TTFT p50 by power mode (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("TTFT p50 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "5_ttft_vs_prompt.png")

# 6. Speedup vs 15W baseline
toks_by_mode = df.groupby(["model", "mode"])["tok_s"].mean().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(14, 6))
for i, cmp_mode in enumerate(["7W", "25W", "MAXN"]):
    if cmp_mode not in toks_by_mode.columns:
        continue
    speedups = []
    for m in MODELS:
        base = toks_by_mode.loc[m, "15W"] if "15W" in toks_by_mode.columns else None
        cmp  = toks_by_mode.loc[m, cmp_mode] if cmp_mode in toks_by_mode.columns else None
        speedups.append(cmp / base if (base and cmp and not pd.isna(base) and not pd.isna(cmp)) else np.nan)
    offs = [-WIDTH, 0, WIDTH]
    bars = ax.bar(np.arange(len(MODELS)) + offs[i], speedups, WIDTH,
                  label=f"{cmp_mode} vs 15W", color=MODE_PAL[cmp_mode],
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, speedups):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(1.0, color="grey", ls="--", lw=1.2, alpha=0.7, label="15W baseline")
ax.set_xticks(np.arange(len(MODELS)))
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Output Throughput Speedup vs 15W: avg over all prompt x gen combos",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Tok/s ratio (vs 15W)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "6_speedup_vs_15w.png")

# 7a/b. Tok/J heatmaps
models_all4  = ["smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "lfm2.5-350m", "lfm2.5-1.2b"]
models_3mode = ["qwen3-0.6b", "llama3.2-1b", "gemma3-1b"]
all_tokj = df["tok_j"].dropna()
vmin, vmax = all_tokj.min(), all_tokj.max()

fig, axes = plt.subplots(4, 5, figsize=(22, 14))
for row, mode in enumerate(["7W", "15W", "25W", "MAXN"]):
    for col, model in enumerate(models_all4):
        ax = axes[row][col]
        sub = df[(df.model == model) & (df["mode"] == mode)]
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            continue
        pivot = sub.pivot_table(index="gen", columns="prompt", values="tok_j", aggfunc="mean")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlGnBu",
                    linewidths=0.5, cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"{MDL[model]} ({mode})", fontsize=8)
        ax.set_xlabel("Prompt (tok)" if row == 3 else "")
        ax.set_ylabel("Gen (tok)" if col == 0 else "")
fig.suptitle("Output Tok/J Heatmap: models at all 4 power modes\n(rows = gen length, cols = prompt length)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "7a_tok_j_heatmap_small_models.png")

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for row, mode in enumerate(["15W", "25W", "MAXN"]):
    for col, model in enumerate(models_3mode):
        ax = axes[row][col]
        sub = df[(df.model == model) & (df["mode"] == mode)]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index="gen", columns="prompt", values="tok_j", aggfunc="mean")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlGnBu",
                    linewidths=0.5, cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"{MDL[model]} ({mode})", fontsize=8)
        ax.set_xlabel("Prompt (tok)" if row == 2 else "")
        ax.set_ylabel("Gen (tok)" if col == 0 else "")
fig.suptitle("Output Tok/J Heatmap: larger models (15W/25W/MAXN only, OOM at 7W)\n"
             "(rows = gen length, cols = prompt length)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "7b_tok_j_heatmap_large_models.png")

# 8. ITL comparison: canonical ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["itl"].mean() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Inter-Token Latency p50: All Power Modes (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("ITL p50 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "8_itl_compare.png")

# 8b. ITL p50 heatmap: models × power modes at canonical ctx=2048, gen=256
itl_matrix = []
for m in MODELS:
    row = []
    for mode in ["7W", "15W", "25W", "MAXN"]:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        row.append(sub["itl"].mean() if not sub.empty else np.nan)
    itl_matrix.append(row)
itl_df = pd.DataFrame(itl_matrix,
                       index=[MDL[m] for m in MODELS],
                       columns=["7W", "15W", "25W", "MAXN"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(itl_df, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "ITL p50 (ms)"},
            ax=ax)
ax.set_title(f"ITL p50 (ms): All Models × All Power Modes\n(ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Power Mode")
ax.set_ylabel("Model")
plt.tight_layout()
save(fig, "8b_itl_heatmap_models_modes.png")

# 9. Request latency (E2E) comparison: canonical ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["rl_avg"].mean() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Request Latency (E2E) p50: All Power Modes (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Request latency p50 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "10_request_latency_compare.png")

# 10. Prefill throughput: gen=256, avg over all prompts
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["prefill"].mean() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Prefill Throughput: All Power Modes (gen={CANONICAL_GEN}, avg over all prompts)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Prefill output tok/s")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "9_prefill_compare.png")

# 11. Tok/s vs power scatter
fig, ax = plt.subplots(figsize=(12, 7))
for model in MODELS:
    for mode, (ls, mk) in MODE_STYLE.items():
        sub = df[(df.model == model) & (df["mode"] == mode)]
        if sub.empty:
            continue
        ax.scatter(sub.power_w, sub.tok_s, color=MODEL_PAL[model],
                   marker=mk, s=60, alpha=0.75, edgecolors="white", linewidths=0.4)
    cx = df[df.model == model]["power_w"].mean()
    cy = df[df.model == model]["tok_s"].mean()
    if not np.isnan(cx) and not np.isnan(cy):
        ax.annotate(MDL[model], xy=(cx, cy), fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc=MODEL_PAL[model], alpha=0.25))
ax.legend(handles=model_legend_handles(MODELS) +
          [mlines.Line2D([], [], color="grey", ls="", marker=mk, label=mode, ms=8)
           for mode, (ls, mk) in MODE_STYLE.items()],
          fontsize=8, ncol=2)
ax.set_title("Output Tok/s vs Power: All Combos and Power Modes\n"
             "(color = model, shape = mode: ^ 7W  o 15W  s 25W  D MAXN)",
             fontweight="bold")
ax.set_xlabel("Power (W)  [VDD_CPU_GPU_CV]")
ax.set_ylabel("Output Tok/s (per user)")
plt.tight_layout()
save(fig, "10_tok_s_vs_power_scatter.png")

# 12. Canonical cell comparison: ctx=2048, gen=256
canonical_bar(CANONICAL_CTX, CANONICAL_GEN,
              fname_prefix="11_canonical_cell_comparison",
              title_suffix="  [canonical standard]")

# 14. SmolLM2-135M spotlight — fixed y-scale across all 3 gen panels
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
model = "smollm2-135m"
for ax, gen_val in zip(axes, [64, 128, 256]):
    for mode, (ls, mk) in MODE_STYLE.items():
        sub = df[(df.model == model) & (df.gen == gen_val) &
                 (df["mode"] == mode)].sort_values("prompt")
        if sub.empty:
            continue
        ax.plot(sub.prompt, sub.tok_j, marker=mk, ls=ls, lw=2,
                color=MODE_PAL[mode], label=mode, ms=8)
    ax.set_title(f"gen={gen_val} tok", fontsize=10)
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Output Tok/J" if ax is axes[0] else "")
    ax.set_xticks(PROMPT_LENGTHS)
    ax.legend(fontsize=9)
fig.suptitle("SmolLM2-135M: Output Tok/J at All 4 Power Modes across gen lengths",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "12_smollm2_135m_tok_j_spotlight.png")


# ════════════════════════════════════════════════════════════════════════════════
# COMPARISON LINE CHARTS  — 2x4 facets with fixed y-scale
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Comparison line charts (fixed y-scale, all metrics × gen lengths) ──")

# Helper for faceted line chart vs prompt
def facet_vs_prompt(gen_val, y_field, y_label, fname):
    faceted_line_chart(df, y_field, y_label, gen_val, fname)

# Helper for faceted line chart vs gen (fix prompt instead)
def facet_vs_gen(ctx_val, y_field, y_label, fname):
    """2×4 facet, sharey=True, x = gen lengths, filtered by prompt=ctx_val."""
    data_sub = df[(df.prompt == ctx_val) & (df["mode"].isin(modes_avail))][y_field].dropna()
    if data_sub.empty:
        return
    y_max = data_sub.max() * 1.15
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    fig.suptitle(f"{y_label} vs Gen Length: All Power Modes (ctx={ctx_val})",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 4][idx % 4]
        sub = df[(df.model == model) & (df.prompt == ctx_val) & (df["mode"].isin(modes_avail))]
        for mode, (ls, mk) in MODE_STYLE.items():
            s = sub[(sub["mode"] == mode)].sort_values("gen")
            if s.empty: continue
            ax.plot(s.gen, s[y_field], marker=mk, ls=ls, lw=2,
                    color=MODE_PAL[mode], label=mode, ms=7)
        ax.set_title(MDL[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Gen (tok)", fontsize=8)
        ax.set_ylabel(y_label if idx % 4 == 0 else "", fontsize=8)
        ax.set_xticks(GEN_LENGTHS)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    save(fig, fname)

# ── Request latency & tok/J charts (22a/e/f/g) ──
for gen_val in [64, 128, 256]:
    prefix = "EF_req_latency" if gen_val < 256 else "22a_request_latency"
    facet_vs_prompt(gen_val, "rl_avg", "Request latency (ms)", f"{prefix}_vs_prompt_gen{gen_val}.png")

for gen_val in [64, 128, 256]:
    for field, label, prefix in [("prefill_tokj", "Prefill tok/J", "22e"), ("total_tokj", "Total tok/J", "22g")]:
        facet_vs_prompt(gen_val, field, label, f"{prefix}_{field}_vs_prompt_gen{gen_val}.png")
    # Decode tok/J vs prompt at gen=256 is F.2c (canonical)
    if gen_val == 256:
        facet_vs_prompt(256, "decode_tokj", "Decode tok/J", "22f_decode_tokj_vs_prompt_gen256.png")

# ── Decode tok/J vs gen length (independent of prompt, better x-axis) ──
for ctx_val in [128, 512, 1024, 2048]:
    facet_vs_gen(ctx_val, "decode_tokj", "Decode tok/J", f"EF_decode_tokj_vs_gen_ctx{ctx_val}.png")

# ── tok/J additional gen lengths (EF) — prefill + total only ──
for gen_val in [64, 128]:
    for field, label in [("prefill_tokj", "Prefill tok/J"), ("total_tokj", "Total tok/J")]:
        facet_vs_prompt(gen_val, field, label, f"EF_{field}_vs_prompt_gen{gen_val}.png")

# ── TTFT (EG) ──
for gen_val in [64, 128, 256]:
    facet_vs_prompt(gen_val, "ttft", "TTFT p50 (ms)", f"EG_ttft_vs_prompt_gen{gen_val}.png")
for ctx_val in [128, 512, 1024, 2048]:
    facet_vs_gen(ctx_val, "ttft", "TTFT p50 (ms)", f"EG_ttft_vs_gen_ctx{ctx_val}.png")

# ── ITL (EH) ──
for gen_val in [64, 128, 256]:
    facet_vs_prompt(gen_val, "itl", "ITL p50 (ms)", f"EH_itl_vs_prompt_gen{gen_val}.png")
for ctx_val in [128, 512, 1024, 2048]:
    facet_vs_gen(ctx_val, "itl", "ITL p50 (ms)", f"EH_itl_vs_gen_ctx{ctx_val}.png")

# ── Prefill throughput (EI) ──
for gen_val in [64, 128, 256]:
    facet_vs_prompt(gen_val, "prefill", "Prefill tok/s", f"EI_prefill_tput_vs_prompt_gen{gen_val}.png")
for ctx_val in [128, 512, 1024, 2048]:
    facet_vs_gen(ctx_val, "prefill", "Prefill tok/s", f"EI_prefill_tput_vs_gen_ctx{ctx_val}.png")

# ── All-models overview (EA) ──
facet_vs_prompt(256, "prefill_j", "Prefill energy (J)", "EA_prefill_energy_vs_prompt_all_models.png")
facet_vs_prompt(256, "rl_avg", "Request latency (ms)", "EA_request_latency_vs_prompt_all_models.png")

# ── Prefill energy vs prompt SmolLM2-135M (Figure 17) ──
fig, ax = plt.subplots(figsize=(10, 6))
model_s = "smollm2-135m"
for mode, (ls, mk) in MODE_STYLE.items():
    sub = df[(df.model == model_s) & (df["mode"] == mode) & (df.gen == 256)].sort_values("prompt")
    if sub.empty: continue
    ax.plot(sub.prompt, sub.prefill_j, marker=mk, ls=ls, lw=2,
            color=MODE_PAL[mode], label=mode, ms=8)
ax.set_title("SmolLM2-135M: Prefill Energy vs Prompt Length (gen=256)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Prompt (tok)")
ax.set_ylabel("Prefill energy (J)")
ax.set_xticks(PROMPT_LENGTHS)
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "E_prefill_energy_vs_prompt.png")

# ── Heatmaps (EG, EH, EI) ──
print("  Generating heatmaps...")
for metric, label, cmap, prefix in [
    ("ttft", "TTFT (ms)", "Reds", "EG_ttft"),
    ("itl", "ITL (ms)", "Oranges", "EH_itl"),
    ("prefill", "Prefill tok/s", "Greens", "EI_prefill_tput"),
]:
    for mode_name in modes_avail:
        sub = df[df["mode"] == mode_name]
        if sub.empty: continue
        pivot = sub.pivot_table(index="gen", columns="prompt", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap=cmap,
                    linewidths=0.5, cbar=True)
        ax.set_title(f"{label} heatmap: {mode_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Prompt (tok)")
        ax.set_ylabel("Gen (tok)")
        plt.tight_layout()
        mode_file = mode_name.lower()
        save(fig, f"{prefix}_heatmap_{mode_file}.png")


# ════════════════════════════════════════════════════════════════════════════════
# APPENDIX E CHARTS  — all 12 prompt x gen combinations
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Appendix E charts (all prompt x gen combinations) ──")

# E1–E8: heatmap grids — tok/s and tok/J for all 4 power modes
for metric, label, fmt_str, cmap in [
    ("tok_s", "Output Tok/s", ".0f", "Blues"),
    ("tok_j", "Output Tok/J", ".1f", "YlGnBu"),
]:
    for mode_name in modes_avail:
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        fig.suptitle(f"{label}: All 12 Prompt x Gen Combinations at {mode_name}",
                     fontsize=14, fontweight="bold")
        for idx, model in enumerate(MODELS):
            ax = axes[idx // 4][idx % 4]
            sub = df[(df.model == model) & (df["mode"] == mode_name)]
            if sub.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            pivot = sub.pivot_table(index="gen", columns="prompt", values=metric, aggfunc="mean")
            sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt_str, cmap=cmap,
                        linewidths=0.5, cbar=False)
            ax.set_title(MDL[model], fontsize=10, fontweight="bold")
            ax.set_xlabel("Prompt (tok)" if idx >= 4 else "")
            ax.set_ylabel("Gen (tok)" if idx % 4 == 0 else "")
        plt.tight_layout()
        save(fig, f"E_{metric}_heatmap_{mode_name.lower()}.png")

# ════════════════════════════════════════════════════════════════════════════════
# APPENDIX F.3 CHARTS  — Request latency (E2E) grouped bars, all combos
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Appendix F.3 charts (request latency, all prompt x gen combos) ──")

def request_latency_bar(ctx_val, gen_val, fname_prefix):
    """Single-panel grouped bar chart: request latency p50 for a specific ctx, gen."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(MODELS))
    for i, mode in enumerate(modes_avail):
        vals = []
        for m in MODELS:
            sub = df[(df.model == m) & (df["mode"] == mode) &
                     (df.prompt == ctx_val) & (df.gen == gen_val)]
            vals.append(sub["rl_avg"].mean() if not sub.empty else np.nan)
        bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                      color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
    ax.set_title(f"Request Latency (E2E) p50: ctx={ctx_val}, gen={gen_val}",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("Request latency p50 (ms)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, f"{fname_prefix}_ctx{ctx_val}_gen{gen_val}.png")

# F.3a–c: varying gen at ctx=2048
for gen_val in GEN_LENGTHS:
    request_latency_bar(CANONICAL_CTX, gen_val, fname_prefix="F3_rlat")

# F.3d–g: varying ctx at gen=256
for ctx_val in PROMPT_LENGTHS:
    request_latency_bar(ctx_val, CANONICAL_GEN, fname_prefix="F3_rlat")

print(f"\nAll charts saved to {OUT_DIR}")
print(f"Main charts:    12")
print(f"Appendix E:     8  (tok/s + tok/J heatmaps for 7W/15W/25W/MAXN)")
print(f"Appendix F.3:   7  (F3_rlat_ctx2048_gen64/128/256, F3_rlat_ctx128/512/1024/2048_gen256)")
comp_count = 2 + 3 + 3*3 + 2 + 2*3 + 3 + 4 + 3 + 4 + 3 + 4 + 2 + 1 + 3*4
print(f"Comparison line: ~{comp_count} (all metrics at all gen/prompt combos)")
