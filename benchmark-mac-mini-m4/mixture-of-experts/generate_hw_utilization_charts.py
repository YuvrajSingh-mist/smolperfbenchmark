#!/usr/bin/env python3
"""Generate GPU-usage and Processor-usage/frequency charts for the Mac Mini M4
MoE benchmark, at the canonical combo (ctx=30720, gen=1024) only.

Source: the raw `powermetrics.log` for each model's canonical combo directory.
Unlike the tok/s, tok/J, latency, and power charts elsewhere in this repo (which
all come from `profile_export_aiperf.json` + the Combined Power rail), these
charts read fields from `powermetrics`' own "Processor usage" and "GPU usage"
sections that no other script in this repo parses:

  - GPU HW active frequency (MHz) + GPU idle residency (%)
  - E-Cluster / P-Cluster HW active frequency (MHz) + idle residency (%)

macOS's `thermal` sampler exposes no numeric per-core temperature (only a
categorical "Current pressure level"), so there is no die-temperature chart
here the way there would be for a `tegrastats`-based benchmark.

Charts saved to artifacts/charts/ (referenced by BLOG.md section 2.6/2.7).
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
LLAMACPP_DIR = ROOT / "artifacts/mac-m4-moe-20260704-0115/llamacpp"
MLXLM_DIR    = ROOT / "artifacts/mac-m4-moe-20260712-1733/mlxlm"
OUT_DIR = ROOT / "artifacts/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CTX = 30720
CANONICAL_GEN = 1024

MODELS = [
    "granite4-h-tiny",
    "lfm2.5-8b-a1b",
    "smallthinker-4b-a0.6b",
    "trinity-nano",
    "gemma4-e2b",
    "gemma4-e4b",
]
MLXLM_MODELS = ["granite4-h-tiny", "lfm2.5-8b-a1b", "trinity-nano"]  # only 3 have MLX weights

MDL_FLAT = {m: m for m in MODELS}
MODEL_PAL = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
MODEL_MARKER = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P"]))

sns.set_theme(style="darkgrid", font_scale=1.05)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── powermetrics "Processor usage" / "GPU usage" section parser ─────────────
GPU_FREQ_RE  = re.compile(r"^GPU HW active frequency:\s+(\d+) MHz")
GPU_IDLE_RE  = re.compile(r"^GPU idle residency:\s+([\d.]+)%")
ECL_FREQ_RE  = re.compile(r"^E-Cluster HW active frequency:\s+(\d+) MHz")
ECL_IDLE_RE  = re.compile(r"^E-Cluster idle residency:\s+([\d.]+)%")
PCL_FREQ_RE  = re.compile(r"^P-Cluster HW active frequency:\s+(\d+) MHz")
PCL_IDLE_RE  = re.compile(r"^P-Cluster idle residency:\s+([\d.]+)%")


def parse_hw_utilization(log_path):
    """One row per powermetrics sample: gpu_freq, gpu_idle, e_freq, e_idle, p_freq, p_idle."""
    rows = []
    cur = {}
    if not log_path.exists():
        return pd.DataFrame(rows)
    with open(log_path, errors="ignore") as f:
        for line in f:
            if line.startswith("*** Sampled system activity"):
                if cur:
                    rows.append(cur)
                cur = {}
                continue
            m = GPU_FREQ_RE.match(line)
            if m: cur["gpu_freq"] = int(m.group(1)); continue
            m = GPU_IDLE_RE.match(line)
            if m: cur["gpu_idle"] = float(m.group(1)); continue
            m = ECL_FREQ_RE.match(line)
            if m: cur["e_freq"] = int(m.group(1)); continue
            m = ECL_IDLE_RE.match(line)
            if m: cur["e_idle"] = float(m.group(1)); continue
            m = PCL_FREQ_RE.match(line)
            if m: cur["p_freq"] = int(m.group(1)); continue
            m = PCL_IDLE_RE.match(line)
            if m: cur["p_idle"] = float(m.group(1)); continue
    if cur:
        rows.append(cur)
    return pd.DataFrame(rows)


def load_canonical(run_dir, models):
    data = {}
    for model in models:
        log_path = run_dir / model / f"gen{CANONICAL_GEN}" / f"ctx{CANONICAL_CTX}" / "powermetrics.log"
        df = parse_hw_utilization(log_path)
        if not df.empty:
            data[model] = df
            print(f"  {model}: {len(df)} samples")
        else:
            print(f"  {model}: no data ({log_path})")
    return data


print("Loading canonical-combo (ctx=30720, gen=1024) powermetrics logs — llama.cpp ...")
lc_data = load_canonical(LLAMACPP_DIR, MODELS)
print("Loading canonical-combo (ctx=30720, gen=1024) powermetrics logs — MLX-LM ...")
mlx_data = load_canonical(MLXLM_DIR, MLXLM_MODELS)


# ═══════════════════════════════════════════════════════════════ GPU Usage ══
# Figure: GPU active frequency over the run, binned into 200 buckets per model
# so ~20-55k raw 50ms samples per model become a readable trend line. X-axis
# is % of run completion (not wall-clock time) so all 6 models overlay on one
# comparable axis regardless of how long their canonical combo took to finish.
print("\n-- GPU Usage: active frequency over run (both backends) --")
fig, ax = plt.subplots(figsize=(12, 6.5))
NBINS = 200


def plot_binned(ax, df, col, model, ls, alpha, label_suffix):
    if df is None or col not in df or df[col].isna().all():
        return
    s = df[col].reset_index(drop=True)
    bin_idx = (np.arange(len(s)) / len(s) * NBINS).astype(int).clip(0, NBINS - 1)
    binned = pd.Series(s.values).groupby(bin_idx).mean()
    xpct = binned.index / NBINS * 100
    ax.plot(xpct, binned.values, lw=2, ls=ls, color=MODEL_PAL[model], alpha=alpha,
            label=f"{MDL_FLAT[model]} ({label_suffix})")


for model in MODELS:
    plot_binned(ax, lc_data.get(model), "gpu_freq", model, "-", 1.0, "llama.cpp")
for model in MLXLM_MODELS:
    plot_binned(ax, mlx_data.get(model), "gpu_freq", model, "--", 0.85, "MLX-LM")
ax.set_xlabel("Run completion (%)")
ax.set_ylabel("GPU HW active frequency (MHz)")
ax.set_title("GPU Active Frequency Over the Canonical-Combo Run (ctx=30720, gen=1024) — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
plt.tight_layout()
save(fig, "gpu_freq_over_run_canonical.png")

# Figure: average GPU active frequency (samples where GPU is actually active,
# i.e. freq > 0) + GPU active residency %, side by side bars, both backends.
print("-- GPU Usage: avg active frequency + active residency bars (both backends) --")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
x = np.arange(len(MODELS))


def freq_residency(data, models):
    avg_freq, active_pct = {}, {}
    for model in models:
        df = data.get(model)
        if df is None or df.empty:
            continue
        active = df.loc[df["gpu_freq"] > 0, "gpu_freq"]
        avg_freq[model] = active.mean() if not active.empty else np.nan
        active_pct[model] = 100 - df["gpu_idle"].mean() if "gpu_idle" in df else np.nan
    return avg_freq, active_pct


avg_freq_lc, active_pct_lc = freq_residency(lc_data, MODELS)
avg_freq_mlx, active_pct_mlx = freq_residency(mlx_data, MLXLM_MODELS)
cols = [MODEL_PAL[m] for m in MODELS]
w = 0.34
vals1_lc  = [avg_freq_lc.get(m, np.nan) for m in MODELS]
vals1_mlx = [avg_freq_mlx.get(m, np.nan) for m in MODELS]
vals2_lc  = [active_pct_lc.get(m, np.nan) for m in MODELS]
vals2_mlx = [active_pct_mlx.get(m, np.nan) for m in MODELS]

bars1 = ax1.bar(x - w / 2, vals1_lc, w, color=cols, edgecolor="white", linewidth=0.8, label="llama.cpp")
bars1m = ax1.bar(x + w / 2, vals1_mlx, w, color=cols, edgecolor="black", linewidth=0.9, hatch="//", alpha=0.85, label="MLX-LM")
for bars in (bars1, bars1m):
    for bar, v in zip(bars, vals1_lc if bars is bars1 else vals1_mlx):
        if not np.isnan(v):
            ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("GPU active frequency (MHz)")
ax1.set_title("Avg GPU Frequency While Active", fontweight="bold", fontsize=12)
ax1.legend(fontsize=8)

bars2 = ax2.bar(x - w / 2, vals2_lc, w, color=cols, edgecolor="white", linewidth=0.8, label="llama.cpp")
bars2m = ax2.bar(x + w / 2, vals2_mlx, w, color=cols, edgecolor="black", linewidth=0.9, hatch="//", alpha=0.85, label="MLX-LM")
for bars in (bars2, bars2m):
    for bar, v in zip(bars, vals2_lc if bars is bars2 else vals2_mlx):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax2.set_ylabel("GPU active residency (%)")
ax2.set_title("GPU Active Residency (% of run GPU is busy)", fontweight="bold", fontsize=12)
ax2.legend(fontsize=8)
# Kept at a full 0-100 baseline on purpose (not zoomed) - bar charts should not
# truncate their baseline. The real finding is that every model keeps the GPU
# saturated ~99-100% of the time at this long-context cell; the exact value
# labels above each bar carry the (noise-level, <0.5pt) spread for anyone who
# wants it.

fig.suptitle(f"GPU Utilization — Canonical Combo (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "gpu_utilization_bars_canonical.png")


# ═══════════════════════════════════════════════════════ Processor Usage ══
# Figure: E-Cluster vs P-Cluster active residency (grouped bar) - how much of
# the canonical-combo run actually lands on efficiency cores vs performance
# cores. Apple's scheduler keeps LLM inference almost entirely on the P-cluster.
print("\n-- Processor Usage: E-Cluster vs P-Cluster active residency (both backends) --")
fig, ax = plt.subplots(figsize=(13, 6.5))
w = 0.19


def cluster_residency(data, models):
    e_active, p_active = {}, {}
    for model in models:
        df = data.get(model)
        if df is None or df.empty:
            continue
        e_active[model] = 100 - df["e_idle"].mean() if "e_idle" in df else np.nan
        p_active[model] = 100 - df["p_idle"].mean() if "p_idle" in df else np.nan
    return e_active, p_active


e_lc, p_lc = cluster_residency(lc_data, MODELS)
e_mlx, p_mlx = cluster_residency(mlx_data, MLXLM_MODELS)
e_active_lc  = [e_lc.get(m, np.nan) for m in MODELS]
p_active_lc  = [p_lc.get(m, np.nan) for m in MODELS]
e_active_mlx = [e_mlx.get(m, np.nan) for m in MODELS]
p_active_mlx = [p_mlx.get(m, np.nan) for m in MODELS]

bars_e  = ax.bar(x - 1.5 * w, e_active_lc, w, label="E-Cluster active % (llama.cpp)", color="#4CAF50", edgecolor="white")
bars_p  = ax.bar(x - 0.5 * w, p_active_lc, w, label="P-Cluster active % (llama.cpp)", color="#F44336", edgecolor="white")
bars_em = ax.bar(x + 0.5 * w, e_active_mlx, w, label="E-Cluster active % (MLX-LM)", color="#4CAF50", edgecolor="black", hatch="//", alpha=0.85)
bars_pm = ax.bar(x + 1.5 * w, p_active_mlx, w, label="P-Cluster active % (MLX-LM)", color="#F44336", edgecolor="black", hatch="//", alpha=0.85)
for bars in (bars_e, bars_p, bars_em, bars_pm):
    for bar in bars:
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Active residency (%)")
ax.set_title(f"E-Cluster vs P-Cluster Active Residency — Canonical Combo (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
             fontweight="bold", fontsize=11.5)
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
save(fig, "cluster_active_residency_canonical.png")

# Figure: average P-Cluster active frequency per model (E-Cluster is active
# too rarely during inference for its average frequency to be meaningful -
# see the residency chart above).
print("-- Processor Usage: avg P-Cluster active frequency (both backends) --")
fig, ax = plt.subplots(figsize=(10, 6.5))


def p_cluster_freq(data, models):
    out = {}
    for model in models:
        df = data.get(model)
        if df is None or df.empty or "p_freq" not in df:
            continue
        active = df.loc[df["p_freq"] > 0, "p_freq"]
        out[model] = active.mean() if not active.empty else np.nan
    return out


p_freq_lc  = p_cluster_freq(lc_data, MODELS)
p_freq_mlx = p_cluster_freq(mlx_data, MLXLM_MODELS)
vals_lc  = [p_freq_lc.get(m, np.nan) for m in MODELS]
vals_mlx = [p_freq_mlx.get(m, np.nan) for m in MODELS]
w = 0.34
bars  = ax.bar(x - w / 2, vals_lc, w, color=cols, edgecolor="white", linewidth=0.8, label="llama.cpp")
barsm = ax.bar(x + w / 2, vals_mlx, w, color=cols, edgecolor="black", linewidth=0.9, hatch="//", alpha=0.85, label="MLX-LM")
for bars_ in (bars, barsm):
    for bar, v in zip(bars_, vals_lc if bars_ is bars else vals_mlx):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("P-Cluster active frequency (MHz)")
ax.set_title(f"Avg P-Cluster (Performance-core) Frequency While Active — Canonical Combo (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
             fontweight="bold", fontsize=11.5)
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "p_cluster_freq_canonical.png")

# Figure: P-Cluster active frequency over the run, same binned-trend treatment
# as the GPU frequency chart above, to show whether clock ramps during prefill
# and holds steady during decode (or stays pinned near max throughout, unlike
# a Jetson-style governor).
print("-- Processor Usage: P-Cluster frequency over run (both backends) --")
fig, ax = plt.subplots(figsize=(12, 6.5))
for model in MODELS:
    plot_binned(ax, lc_data.get(model), "p_freq", model, "-", 1.0, "llama.cpp")
for model in MLXLM_MODELS:
    plot_binned(ax, mlx_data.get(model), "p_freq", model, "--", 0.85, "MLX-LM")
ax.set_xlabel("Run completion (%)")
ax.set_ylabel("P-Cluster HW active frequency (MHz)")
ax.set_title("P-Cluster Active Frequency Over the Canonical-Combo Run (ctx=30720, gen=1024) — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
plt.tight_layout()
save(fig, "p_cluster_freq_over_run_canonical.png")


# ═══════════════════════════════════════════════════════════════ RSS Usage ══
# rss.log: one `ps -o rss=` sample (KB) per line, taken every 50ms for the
# server process's whole combo window (see start_combo_rss in benchmark-moe.sh).
# This is model weights + KV-cache + activations - it grows with context, not
# a fixed "model size" figure. Canonical combo (ctx=30720) is where that KV-cache
# growth is largest, so it's the most informative cell to chart it at.
print("\nLoading canonical-combo rss.log files — llama.cpp ...")


def load_rss(run_dir, models):
    data = {}
    for model in models:
        rss_path = run_dir / model / f"gen{CANONICAL_GEN}" / f"ctx{CANONICAL_CTX}" / "rss.log"
        if not rss_path.exists():
            print(f"  {model}: no data ({rss_path})")
            continue
        try:
            vals = [int(line) / 1024.0 for line in open(rss_path) if line.strip()]  # KB -> MB
        except Exception:
            continue
        if vals:
            data[model] = np.array(vals)
            print(f"  {model}: {len(vals)} samples, peak {max(vals):.0f} MB")
    return data


lc_rss  = load_rss(LLAMACPP_DIR, MODELS)
print("Loading canonical-combo rss.log files — MLX-LM ...")
mlx_rss = load_rss(MLXLM_DIR, MLXLM_MODELS)

# Figure: RSS over the run, binned into 200 buckets per model, same normalized
# "% of run completion" x-axis as the GPU/P-cluster frequency charts above.
print("-- RSS Usage: memory over run (both backends) --")
fig, ax = plt.subplots(figsize=(12, 6.5))


def plot_binned_arr(ax, s, model, ls, alpha, label_suffix):
    if s is None or len(s) == 0:
        return
    bin_idx = (np.arange(len(s)) / len(s) * NBINS).astype(int).clip(0, NBINS - 1)
    binned = pd.Series(s).groupby(bin_idx).mean()
    xpct = binned.index / NBINS * 100
    ax.plot(xpct, binned.values, lw=2, ls=ls, color=MODEL_PAL[model], alpha=alpha,
            label=f"{MDL_FLAT[model]} ({label_suffix})")


for model in MODELS:
    plot_binned_arr(ax, lc_rss.get(model), model, "-", 1.0, "llama.cpp")
for model in MLXLM_MODELS:
    plot_binned_arr(ax, mlx_rss.get(model), model, "--", 0.85, "MLX-LM")
ax.set_xlabel("Run completion (%)")
ax.set_ylabel("RSS (MB)")
ax.set_title("Server Process RSS Over the Canonical-Combo Run (ctx=30720, gen=1024) — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
plt.tight_layout()
save(fig, "rss_over_run_canonical.png")

# Figure: peak RSS per model at the canonical combo specifically (distinct from
# Appendix I.2's i2g_peak_ram.png, which takes the max across all 18 combos -
# this one isolates the single largest-context cell). Both backends.
print("-- RSS Usage: peak RSS bar, canonical combo (both backends) --")
fig, ax = plt.subplots(figsize=(10, 6.5))
peak_rss_lc  = [lc_rss[m].max() if m in lc_rss and len(lc_rss[m]) else np.nan for m in MODELS]
peak_rss_mlx = [mlx_rss[m].max() if m in mlx_rss and len(mlx_rss[m]) else np.nan for m in MODELS]
w = 0.34
bars  = ax.bar(x - w / 2, peak_rss_lc, w, color=cols, edgecolor="white", linewidth=0.8, label="llama.cpp")
barsm = ax.bar(x + w / 2, peak_rss_mlx, w, color=cols, edgecolor="black", linewidth=0.9, hatch="//", alpha=0.85, label="MLX-LM")
for bars_ in (bars, barsm):
    vals = peak_rss_lc if bars_ is bars else peak_rss_mlx
    for bar, v in zip(bars_, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} MB", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Peak RSS (MB)")
ax.set_title(f"Peak RSS at the Canonical Combo (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "rss_peak_canonical.png")

print(f"\nAll GPU/Processor/RSS utilization charts saved to {OUT_DIR}")
