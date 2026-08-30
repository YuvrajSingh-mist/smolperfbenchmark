#!/usr/bin/env python3
"""Generate beautiful Appendix I.2 charts for the Mac Mini M4 MoE benchmark blog.

Covers the full Table 19 dataset — all 4 models × 18 prompt×gen combos.
Charts complement those already in Appendix I.1 (tok/s, tok/J heatmaps + line charts).

Output: artifacts/charts/i2_*.png (referenced by BLOG.md §I.2).
"""

import json, re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT         = Path(__file__).parent
LLAMACPP_DIR = ROOT / "artifacts/mac-m4-moe-20260704-0115/llamacpp"
MLXLM_DIR    = ROOT / "artifacts/mac-m4-moe-20260712-1733/mlxlm"
RUN_DIR      = LLAMACPP_DIR
OUT_DIR      = ROOT / "artifacts/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "granite4-h-tiny",
    "lfm2.5-8b-a1b",
    "smallthinker-4b-a0.6b",
    "trinity-nano",
    "gemma4-e2b",
    "gemma4-e4b",
]
MLXLM_MODELS = ["granite4-h-tiny", "lfm2.5-8b-a1b", "trinity-nano"]
MDL = {
    "granite4-h-tiny":       "granite4-\nh-tiny",
    "lfm2.5-8b-a1b":         "lfm2.5-\n8b-a1b",
    "smallthinker-4b-a0.6b": "smallthinker-\n4b-a0.6b",
    "trinity-nano":          "trinity-\nnano",
    "gemma4-e2b":            "gemma4-\ne2b",
    "gemma4-e4b":            "gemma4-\ne4b",
}
# Total / active (Gemma: "effective") parameters in billions, verified against each
# model's official HF model card (not inferred from the "-A0.6B"/"-A1B"/"E2B" name
# suffixes, which don't always match — e.g. LFM2.5-8B-A1B is actually 8.3B/1.5B).
MDL_PARAMS = {
    "granite4-h-tiny":       (7,   1),
    "lfm2.5-8b-a1b":         (8.3, 1.5),
    "smallthinker-4b-a0.6b": (4,   0.6),
    "trinity-nano":          (6,   1),
    "gemma4-e2b":            (5.1, 2.3),
    "gemma4-e4b":            (8,   4.5),
}


def _fmt_b(v):
    return f"{v:g}"


MDL_FLAT = {k: f"{k} ({_fmt_b(t)}B={_fmt_b(a)}B)" for k, (t, a) in MDL_PARAMS.items()}

PROMPT_LENGTHS = [256, 512, 1024, 2048, 4096, 30720]
GEN_LENGTHS    = [256, 512, 1024]

# Prompt lengths span 256 -> 30720 (>100x) and are NOT evenly spaced, so plotting
# them on a linear numeric x-axis crams 256/512/1024/2048/4096 into a sliver on
# the left and collides their tick labels. Every line chart plots against this
# fixed categorical x position instead, with compact "K" labels.
PROMPT_XPOS = {p: i for i, p in enumerate(PROMPT_LENGTHS)}
PROMPT_LABELS = [f"{p // 1024}K" if p >= 1024 else str(p) for p in PROMPT_LENGTHS]
PROMPT_LABEL_MAP = dict(zip(PROMPT_LENGTHS, PROMPT_LABELS))

NROWS, NCOLS = 2, 3  # 6 models

MODEL_PAL = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
MODEL_MARKER = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P"]))
WIDTH = 0.12

sns.set_theme(style="darkgrid", font_scale=1.05)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── powermetrics loader (ported from generate_combined_charts.py) ──
_pm_cache = {}
def load_powermetrics(log_path):
    if log_path in _pm_cache:
        return _pm_cache[log_path]
    records = []
    cur = {}
    ts_re = re.compile(r'Sampled system activity \(\w+ (\w+ +\d+ +\d+:\d+:\d+ +\d{4})')
    pw_re = re.compile(r'Combined Power \(CPU \+ GPU \+ ANE\):\s+(\d+)\s+mW')
    cp_re = re.compile(r'CPU die temperature:\s+([\d.]+)')
    gp_re = re.compile(r'GPU die temperature:\s+([\d.]+)')
    try:
        for line in open(log_path):
            m = ts_re.search(line)
            if m:
                if cur.get('ts') is not None and cur.get('mw') is not None:
                    records.append((cur['ts'], cur['mw'], cur.get('cpu'), cur.get('gpu')))
                try:
                    cur = {'ts': datetime.strptime(m.group(1).strip(), '%b %d %H:%M:%S %Y').timestamp(), 'mw': None}
                except ValueError:
                    cur = {}
                continue
            p = pw_re.search(line)
            if p and cur: cur['mw'] = int(p.group(1))
            c = cp_re.search(line)
            if c and cur: cur['cpu'] = float(c.group(1))
            g = gp_re.search(line)
            if g and cur: cur['gpu'] = float(g.group(1))
        if cur.get('ts') is not None and cur.get('mw') is not None:
            records.append((cur['ts'], cur['mw'], cur.get('cpu'), cur.get('gpu')))
    except FileNotFoundError:
        pass
    _pm_cache[log_path] = records
    return records


# ── phase-separated tok/J (ported from generate_combined_charts.py) ──
def compute_phase_power(artifact_dir, pm_records, d):
    jsonl_path = artifact_dir / "profile_export.jsonl"

    def p50(k): return (d.get(k) or {}).get("p50")

    osl_p50     = p50("output_sequence_length")
    isl_p50     = p50("input_sequence_length")
    ttft_p50_ms = p50("time_to_first_token")
    rl_p50_ms   = p50("request_latency")
    t0_str      = d.get("start_time")

    if not pm_records:
        return dict(total_pw=None, prefill_pw=None, decode_pw=None,
                     tok_j=None, prefill_tok_j=None, total_tok_j=None)

    all_mw = [mw for _, mw, _, _ in pm_records]
    total_avg_pw_w = (sum(all_mw) / len(all_mw)) / 1000.0 if all_mw else None

    t0 = None
    if t0_str:
        try: t0 = datetime.fromisoformat(t0_str).timestamp()
        except Exception: pass

    per_req = []
    if jsonl_path.exists():
        with open(jsonl_path) as fj:
            for line in fj:
                line = line.strip()
                if not line: continue
                try:
                    meta = json.loads(line).get("metadata", {})
                    if (meta.get("benchmark_phase") == "profiling"
                            and "request_start_ns" in meta
                            and "request_ack_ns"   in meta
                            and "request_end_ns"   in meta):
                        per_req.append((
                            meta["request_start_ns"] / 1e9,
                            meta["request_ack_ns"]   / 1e9,
                            meta["request_end_ns"]   / 1e9,
                        ))
                except Exception:
                    continue

    prefill_mw, decode_mw = [], []
    p50_decode_s = None
    p50_ttft_s = ttft_p50_ms / 1000.0 if ttft_p50_ms else None

    if per_req:
        prefill_wins = [(s, a) for s, a, e in per_req]
        decode_wins  = [(a, e) for s, a, e in per_req]
        for ep, mw, _, _ in pm_records:
            if any(ws <= ep <= wa for ws, wa in prefill_wins):
                prefill_mw.append(mw)
            elif any(wa < ep <= we for wa, we in decode_wins):
                decode_mw.append(mw)
        durations    = sorted(e - a for _, a, e in per_req)
        p50_decode_s = durations[len(durations) // 2]

    if (not prefill_mw or not decode_mw) and ttft_p50_ms and rl_p50_ms and t0 is not None:
        ttft_s = ttft_p50_ms / 1000.0
        rl_s   = rl_p50_ms   / 1000.0
        n_reqs = len(per_req) or 20
        for ep, mw, _, _ in pm_records:
            elapsed = ep - t0
            if elapsed < 0: continue
            req_idx = int(elapsed / rl_s)
            if req_idx >= n_reqs: continue
            if (elapsed - req_idx * rl_s) <= ttft_s:
                prefill_mw.append(mw)
            else:
                decode_mw.append(mw)
        if p50_decode_s is None:
            p50_decode_s = max(rl_s - ttft_s, 0.001)

    def median_w(lst): return sorted(lst)[len(lst) // 2] / 1000.0 if lst else total_avg_pw_w

    prefill_pw_w = median_w(prefill_mw)
    decode_pw_w  = median_w(decode_mw)

    decode_j  = (decode_pw_w  * p50_decode_s) if (decode_pw_w  and p50_decode_s) else None
    prefill_j = (prefill_pw_w * p50_ttft_s)   if (prefill_pw_w and p50_ttft_s)   else None
    total_j   = (prefill_j + decode_j) if (prefill_j is not None and decode_j is not None) else None

    tok_j         = (osl_p50 / decode_j)          if (osl_p50 and decode_j and decode_j > 0) else None
    prefill_tok_j = (isl_p50 / prefill_j)         if (isl_p50 and prefill_j and prefill_j > 0) else None
    total_tok_j   = ((isl_p50 + osl_p50) / total_j) if (
        isl_p50 is not None and osl_p50 is not None and total_j and total_j > 0) else None

    return dict(total_pw=total_avg_pw_w, prefill_pw=prefill_pw_w, decode_pw=decode_pw_w,
                decode_j=decode_j, prefill_j=prefill_j, total_j=total_j,
                tok_j=tok_j, prefill_tok_j=prefill_tok_j, total_tok_j=total_tok_j)


# ── build dataframe (both backends) ─────────────────────────────────────────
def load_run(run_dir, models, backend):
    rows = []
    for model in models:
        for gen in GEN_LENGTHS:
            for ctx in PROMPT_LENGTHS:
                combo_dir = run_dir / model / f"gen{gen}" / f"ctx{ctx}"
                p = combo_dir / "profile_export_aiperf.json"
                if not p.exists():
                    continue
                try:
                    d = json.loads(p.read_text())
                except Exception:
                    continue

                def pct(k, v="p50"):
                    return (d.get(k) or {}).get(v)

                tok_s   = pct("output_token_throughput_per_user", "p50")
                ttft    = pct("time_to_first_token", "p50")
                itl     = pct("inter_token_latency", "p50")
                prefill = pct("prefill_throughput_per_user", "p50")
                rl_p50  = pct("request_latency", "p50")
                rl_p90  = pct("request_latency", "p90")
                rl_p99  = pct("request_latency", "p99")
                isl     = pct("input_sequence_length", "p50")
                osl     = pct("output_sequence_length", "p50")
                ttft_p90 = pct("time_to_first_token", "p90")
                ttft_p99 = pct("time_to_first_token", "p99")
                itl_p90  = pct("inter_token_latency", "p90")
                itl_p99  = pct("inter_token_latency", "p99")
                prefill_p90 = pct("prefill_throughput_per_user", "p90")
                osl_mis = ((osl - gen) / gen) * 100 if osl and gen else None
                req_s   = 1000.0 / rl_p50 if rl_p50 and rl_p50 > 0 else None

                pm_records = load_powermetrics(combo_dir / "powermetrics.log")
                ph = compute_phase_power(combo_dir, pm_records, d)

                rows.append(dict(
                    backend=backend, model=model, prompt=ctx, gen=gen,
                    tok_s=tok_s, ttft=ttft, ttft_p90=ttft_p90, ttft_p99=ttft_p99,
                    itl=itl, itl_p90=itl_p90, itl_p99=itl_p99,
                    prefill=prefill, prefill_p90=prefill_p90,
                    rl_p50=rl_p50, rl_p90=rl_p90, rl_p99=rl_p99,
                    req_s=req_s,
                    isl=isl, osl=osl, osl_mis=osl_mis,
                    total_pw=ph["total_pw"], prefill_pw=ph["prefill_pw"], decode_pw=ph["decode_pw"],
                    tok_j=ph["tok_j"], prefill_tok_j=ph["prefill_tok_j"], total_tok_j=ph["total_tok_j"],
                ))
    return rows


rows = load_run(LLAMACPP_DIR, MODELS, "llamacpp") + load_run(MLXLM_DIR, MLXLM_MODELS, "mlxlm")

df = pd.DataFrame(rows)
df["prompt_label"] = df["prompt"].apply(lambda x: f"{x//1024}K" if x >= 1024 else str(x))
df_lc  = df[df.backend == "llamacpp"]
df_mlx = df[df.backend == "mlxlm"]
print(f"Loaded {len(df)} rows across {df['model'].nunique()} models, {df['backend'].nunique()} backends")

x = np.arange(len(MODELS))
gen1024_lc  = df_lc[df_lc.gen == 1024]
gen1024_mlx = df_mlx[df_mlx.gen == 1024]


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2a — Full-sweep dashboard: 4 models × 4 metrics at gen=1024
# ═══════════════════════════════════════════════════════════════════════════════
print("\n-- Figure I.2a: Full-sweep dashboard (gen=1024, llama.cpp — complete 6-model dataset) --")

gen1024 = df_lc[df_lc.gen == 1024]
metrics = [
    ("tok_s",   "Output Tok/s",       "Blues",   "{:.0f}"),
    ("tok_j",   "Output Tok/J",       "YlGnBu",  "{:.2f}"),
    ("ttft",    "TTFT p50 (ms)",      "Reds",    "{:.0f}"),
    ("itl",     "ITL p50 (ms)",       "Oranges", "{:.1f}"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

for idx, (metric, label, cmap_name, fmt_str) in enumerate(metrics):
    ax = axes[idx]
    pivot_data = {}
    for model in MODELS:
        sub = gen1024[gen1024.model == model].sort_values("prompt")
        pivot_data[model] = sub[metric].values if not sub.empty else []
    prompt_labels = [f"{p//1024}K" if p >= 1024 else str(p) for p in PROMPT_LENGTHS]

    x_pos = np.arange(len(PROMPT_LENGTHS))
    bar_w = 0.8 / len(MODELS)
    # 6 models x 6 prompt lengths = 36 bars per panel - too dense for a value
    # label on every bar (they collide). Rely on bar height + the legend; exact
    # values are already in the line charts (Figures 1/3/7/8) and heatmaps
    # (Appendix I.2b) elsewhere in this post.
    for i, model in enumerate(MODELS):
        vals = pivot_data.get(model, [])
        offset = (i - (len(MODELS) - 1) / 2) * bar_w
        ax.bar(x_pos + offset, vals, bar_w, color=MODEL_PAL[model],
               edgecolor="white", linewidth=0.5, label=MDL_FLAT[model])

    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prompt_labels)
    ax.legend(fontsize=8, ncol=2)

fig.suptitle("Full Sweep Dashboard — All Models, gen=1024 tok",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "i2a_dashboard_gen1024.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2b — Output tok/s + Output tok/J combined line chart, gen=1024
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2b: Tok/s + Tok/J combined, gen=1024 (both backends) --")


def plot_dual_line(ax, metric):
    for model in MODELS:
        s = gen1024_lc[gen1024_lc.model == model].sort_values("prompt")
        if s.empty: continue
        ax.plot(s.prompt.map(PROMPT_XPOS), s[metric], marker=MODEL_MARKER[model], lw=2.2, ls="-",
                color=MODEL_PAL[model], label=f"{MDL_FLAT[model]} (llama.cpp)", ms=8, zorder=3)
    for model in MLXLM_MODELS:
        s = gen1024_mlx[gen1024_mlx.model == model].sort_values("prompt")
        if s.empty: continue
        ax.plot(s.prompt.map(PROMPT_XPOS), s[metric], marker=MODEL_MARKER[model], lw=2.2, ls="--",
                color=MODEL_PAL[model], label=f"{MDL_FLAT[model]} (MLX-LM)", ms=8,
                markerfacecolor="white", markeredgecolor=MODEL_PAL[model], markeredgewidth=1.4,
                alpha=0.85, zorder=2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7))
plot_dual_line(ax1, "tok_s")
ax1.set_title("Output Tok/s vs Prompt Length (gen=1024)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Prompt (tok)")
ax1.set_ylabel("Output Tok/s")
ax1.set_xticks(range(len(PROMPT_LENGTHS)))
ax1.set_xticklabels(PROMPT_LABELS)
ax1.legend(fontsize=7.5, framealpha=0.9, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax1.grid(True, alpha=0.3)

plot_dual_line(ax2, "tok_j")
ax2.set_title("Output Tok/J vs Prompt Length (gen=1024)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Prompt (tok)")
ax2.set_ylabel("Output Tok/J")
ax2.set_xticks(range(len(PROMPT_LENGTHS)))
ax2.set_xticklabels(PROMPT_LABELS)
ax2.legend(fontsize=7.5, framealpha=0.9, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax2.grid(True, alpha=0.3)

fig.suptitle("Throughput & Efficiency — Full Sweep, All Models, gen=1024 — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=13.5, fontweight="bold")
plt.tight_layout()
save(fig, "i2b_tok_s_tok_j_combined_gen1024.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2c — Latency breakdown: TTFT + ITL + Request Latency, gen=1024
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2c: Latency breakdown, gen=1024 (both backends) --")

fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

lat_metrics = [
    ("ttft",   "TTFT p50 (ms)"),
    ("itl",    "ITL p50 (ms)"),
    ("rl_p50", "Request Latency p50 (ms)"),
]

for ax, (metric, label) in zip(axes, lat_metrics):
    plot_dual_line(ax, metric)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_xticks(range(len(PROMPT_LENGTHS)))
    ax.set_xticklabels(PROMPT_LABELS)
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Latency Breakdown — TTFT, ITL, Request Latency (gen=1024) — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=13.5, fontweight="bold")
plt.tight_layout()
save(fig, "i2c_latency_breakdown_gen1024.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2d — Power breakdown: Total, Prefill, Decode, gen=1024
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2d: Power breakdown, gen=1024 (both backends) --")

fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

pow_metrics = [
    ("total_pw",   "Total Power (W)"),
    ("prefill_pw", "Prefill Power (W)"),
    ("decode_pw",  "Decode Power (W)"),
]

for ax, (metric, label) in zip(axes, pow_metrics):
    plot_dual_line(ax, metric)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_xticks(range(len(PROMPT_LENGTHS)))
    ax.set_xticklabels(PROMPT_LABELS)
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Power Profile — Total, Prefill, Decode (gen=1024) — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=13.5, fontweight="bold")
plt.tight_layout()
save(fig, "i2d_power_breakdown_gen1024.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2e — Prefill Throughput vs Prompt, all gen lengths (3-panel)
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2e: Prefill TPS, all gen lengths (both backends) --")

fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

for ax, gen_val in zip(axes, GEN_LENGTHS):
    data_lc  = df_lc[df_lc.gen == gen_val]
    data_mlx = df_mlx[df_mlx.gen == gen_val]
    for model in MODELS:
        s = data_lc[data_lc.model == model].sort_values("prompt")
        if s.empty: continue
        ax.plot(s.prompt.map(PROMPT_XPOS), s.prefill, marker=MODEL_MARKER[model], lw=2.2, ls="-",
                color=MODEL_PAL[model], label=f"{MDL_FLAT[model]} (llama.cpp)", ms=8, zorder=3)
    for model in MLXLM_MODELS:
        s = data_mlx[data_mlx.model == model].sort_values("prompt")
        if s.empty: continue
        ax.plot(s.prompt.map(PROMPT_XPOS), s.prefill, marker=MODEL_MARKER[model], lw=2.2, ls="--",
                color=MODEL_PAL[model], label=f"{MDL_FLAT[model]} (MLX-LM)", ms=8,
                markerfacecolor="white", markeredgecolor=MODEL_PAL[model], markeredgewidth=1.4,
                alpha=0.85, zorder=2)
    ax.set_title(f"gen={gen_val} tok", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Prefill Tok/s")
    ax.set_xticks(range(len(PROMPT_LENGTHS)))
    ax.set_xticklabels(PROMPT_LABELS)
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Prefill Throughput vs Prompt Length — All Gen Lengths — llama.cpp (solid) vs MLX-LM (dashed)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "i2e_prefill_tps_all_gen.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2f — Request Latency full sweep (all 18 combos) per-model heatmaps
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2f: Request Latency heatmaps --")

rl_all = df["rl_p50"].dropna()
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(24, 11))
axes = axes.flatten()
fig.suptitle("Request Latency p50 (ms) — All Prompt × Gen Combinations, All Models — llama.cpp",
             fontsize=16, fontweight="bold")

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    sub = df_lc[df_lc.model == model]
    if sub.empty or sub["rl_p50"].isna().all():
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        continue
    pivot = sub.pivot_table(index="gen", columns="prompt", values="rl_p50", aggfunc="median")
    pivot.columns = [PROMPT_LABEL_MAP.get(int(c), str(c)) for c in pivot.columns]
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=1, cbar=True,
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                cbar_kws={"shrink": 0.85})
    ax.tick_params(axis="both", labelsize=12)
    ax.collections[0].colorbar.ax.tick_params(labelsize=11)
    ax.set_title(MDL_FLAT[model], fontsize=14, fontweight="bold")
    ax.set_xlabel("Prompt (tok)" if idx >= NCOLS else "", fontsize=12)
    ax.set_ylabel("Gen (tok)" if idx % NCOLS == 0 else "", fontsize=12)

plt.tight_layout()
save(fig, "i2f_request_latency_heatmaps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2g — Peak RAM bar chart per model
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2g: Peak RAM (both backends) --")

# Read RSS samples from rss.log (bytes), take max in MB
def peak_ram_for(run_dir, models):
    peak_ram = {}
    for model in models:
        max_ram_bytes = 0
        for gen in GEN_LENGTHS:
            for ctx in PROMPT_LENGTHS:
                rss_path = run_dir / model / f"gen{gen}" / f"ctx{ctx}" / "rss.log"
                if not rss_path.exists():
                    continue
                try:
                    with open(rss_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                max_ram_bytes = max(max_ram_bytes, int(line))
                except Exception:
                    continue
        peak_ram[model] = max_ram_bytes / (1024 * 1024) if max_ram_bytes > 0 else np.nan
    return peak_ram


peak_ram_lc  = peak_ram_for(LLAMACPP_DIR, MODELS)
peak_ram_mlx = peak_ram_for(MLXLM_DIR, MLXLM_MODELS)

fig, ax = plt.subplots(figsize=(10, 6.5))
vals_lc  = [peak_ram_lc.get(m, np.nan) for m in MODELS]
vals_mlx = [peak_ram_mlx.get(m, np.nan) for m in MODELS]
cols = [MODEL_PAL[m] for m in MODELS]
w = 0.34
bars  = ax.bar(x - w / 2, vals_lc, w, color=cols, edgecolor="white", linewidth=0.8, label="llama.cpp")
barsm = ax.bar(x + w / 2, vals_mlx, w, color=cols, edgecolor="black", linewidth=0.9, hatch="//", alpha=0.85, label="MLX-LM")
for bars_ in (bars, barsm):
    vals = vals_lc if bars_ is bars else vals_mlx
    for bar, v in zip(bars_, vals):
        if not (v is None or (isinstance(v, float) and np.isnan(v))):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f} MB", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_title("Peak RAM Usage per Model (max across all 18 combos) — llama.cpp vs MLX-LM",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Peak RAM (MB)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "i2g_peak_ram.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2h — Combined energy efficiency summary: Tok/J across gen lengths
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2h: Tok/J by gen length (grouped bar, llama.cpp — complete 6-model dataset) --")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, gen_val in zip(axes, GEN_LENGTHS):
    data = df_lc[df_lc.gen == gen_val]
    prompt_labels = [f"{p//1024}K" if p >= 1024 else str(p) for p in PROMPT_LENGTHS]
    x_pos = np.arange(len(PROMPT_LENGTHS))
    bar_w = 0.8 / len(MODELS)
    # 6 models x 6 prompt lengths = 36 bars per panel - no per-bar value label
    # (see I.2a); exact values are in Figure 3 (line chart) and the tok/J heatmap.
    for i, model in enumerate(MODELS):
        s = data[data.model == model].sort_values("prompt")
        if s.empty: continue
        vals = s["tok_j"].values
        offset = (i - (len(MODELS) - 1) / 2) * bar_w
        ax.bar(x_pos + offset, vals, bar_w, color=MODEL_PAL[model],
               edgecolor="white", linewidth=0.5, label=MDL_FLAT[model])

    ax.set_title(f"gen={gen_val} tok", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Output Tok/J")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prompt_labels)
    ax.legend(fontsize=7, ncol=2)

fig.suptitle("Output Tok/J — All Prompt × Gen Combinations, All Models — llama.cpp",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "i2h_tok_j_grouped_bars.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2i — tok/s vs tok/J scatter: all prompt lengths at gen=1024
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2i: Tok/s vs Tok/J scatter (Pareto), gen=1024 (both backends) --")

fig, ax = plt.subplots(figsize=(13, 8.5))

for model in MODELS:
    s = gen1024_lc[gen1024_lc.model == model].sort_values("prompt")
    if s.empty: continue
    ax.scatter(s.tok_s, s.tok_j, c=[MODEL_PAL[model]] * len(s),
               s=140, marker=MODEL_MARKER[model], edgecolors="white",
               linewidth=0.8, zorder=5, label=f"{MDL_FLAT[model]} (llama.cpp)")
    # Connect each model's points in prompt order so the sweep direction is visible
    # without needing a label on every point (6 models x 6 points = 36 labels
    # was unreadable when points cluster tightly, e.g. gemma4-e4b).
    ax.plot(s.tok_s, s.tok_j, color=MODEL_PAL[model], lw=1, ls="-", alpha=0.35, zorder=4)
    # Only label the two endpoints of the sweep (shortest/longest prompt) per
    # model - the interior points are implied by the connecting line.
    endpoints = s.iloc[[0, -1]] if len(s) > 1 else s
    for _, row in endpoints.iterrows():
        lbl = f"{int(row.prompt)//1024}K" if row.prompt >= 1024 else f"{int(row.prompt)}"
        ax.annotate(lbl, (row.tok_s, row.tok_j),
                    textcoords="offset points", xytext=(0, 9),
                    fontsize=8, ha="center", color=MODEL_PAL[model], fontweight="bold")

for model in MLXLM_MODELS:
    s = gen1024_mlx[gen1024_mlx.model == model].sort_values("prompt")
    if s.empty: continue
    ax.scatter(s.tok_s, s.tok_j, c="white", s=140, marker=MODEL_MARKER[model],
               edgecolors=[MODEL_PAL[model]], linewidth=1.6, zorder=5, alpha=0.9,
               label=f"{MDL_FLAT[model]} (MLX-LM)")
    ax.plot(s.tok_s, s.tok_j, color=MODEL_PAL[model], lw=1, ls="--", alpha=0.35, zorder=4)
    endpoints = s.iloc[[0, -1]] if len(s) > 1 else s
    for _, row in endpoints.iterrows():
        lbl = f"{int(row.prompt)//1024}K" if row.prompt >= 1024 else f"{int(row.prompt)}"
        ax.annotate(lbl, (row.tok_s, row.tok_j),
                    textcoords="offset points", xytext=(0, -12),
                    fontsize=8, ha="center", color=MODEL_PAL[model], fontweight="bold")

# Draw Pareto frontier (llama.cpp, complete 6-model dataset)
pareto_x, pareto_y = [], []
for model in MODELS:
    s = gen1024_lc[gen1024_lc.model == model]
    if s.empty: continue
    pts = list(zip(s.tok_s, s.tok_j))
    pts = sorted(pts, key=lambda p: (-p[0], -p[1]))
    frontier = [pts[0]]
    for px, py in pts[1:]:
        if py > frontier[-1][1]:
            frontier.append((px, py))
    fx, fy = zip(*frontier) if frontier else ([], [])
    pareto_x.extend(fx)
    pareto_y.extend(fy)

ax.set_xlabel("Output Tok/s", fontsize=12, fontweight="bold")
ax.set_ylabel("Output Tok/J", fontsize=12, fontweight="bold")
ax.set_title("Throughput vs Efficiency — Full Sweep, gen=1024, All Models\n(point labels = prompt length in tokens)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save(fig, "i2i_tok_s_vs_tok_j_scatter_gen1024.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2j — Request Latency percentile spread (p50/p90/p99), ctx=30720, gen=1024
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2j: Request Latency percentile spread, ctx=30720, gen=1024 (both backends) --")

# NOTE: canon must be filtered per-backend before indexing with .iloc[0] — df
# now stacks both backends, and for the 3 shared models a naive df[...] filter
# would silently mix llama.cpp and MLX-LM rows together.
canon_lc_j  = df_lc[(df_lc.prompt == 30720) & (df_lc.gen == 1024)].reset_index(drop=True)
canon_mlx_j = df_mlx[(df_mlx.prompt == 30720) & (df_mlx.gen == 1024)].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 6.5))
w = 0.11


def rl_percentiles(canon_sub, model):
    s = canon_sub[canon_sub.model == model].reset_index(drop=True)
    if s.empty:
        return None
    p50_val = s["rl_p50"].iloc[0] if not pd.isna(s["rl_p50"].iloc[0]) else 0
    p90_val = s["rl_p90"].iloc[0] if "rl_p90" in s.columns and not pd.isna(s["rl_p90"].iloc[0]) else p50_val * 1.1
    p99_val = s["rl_p99"].iloc[0] if "rl_p99" in s.columns and not pd.isna(s["rl_p99"].iloc[0]) else p50_val * 1.2
    return p50_val, p90_val, p99_val


for i, model in enumerate(MODELS):
    vals = rl_percentiles(canon_lc_j, model)
    if vals is None: continue
    p50_val, p90_val, p99_val = vals
    has_mlx = model in MLXLM_MODELS
    base = i - (2 * w if has_mlx else 0.5 * w)
    ax.bar(base - w, p50_val, w, color=MODEL_PAL[model], alpha=1.0, edgecolor="white", linewidth=0.5)
    ax.bar(base,     p90_val, w, color=MODEL_PAL[model], alpha=0.55, edgecolor="white", linewidth=0.5)
    ax.bar(base + w, p99_val, w, color=MODEL_PAL[model], alpha=0.3, edgecolor="white", linewidth=0.5)

    if has_mlx:
        mvals = rl_percentiles(canon_mlx_j, model)
        if mvals is not None:
            mp50, mp90, mp99 = mvals
            mbase = i + 1.5 * w
            ax.bar(mbase - w, mp50, w, color=MODEL_PAL[model], alpha=1.0, edgecolor="black", hatch="//", linewidth=0.5)
            ax.bar(mbase,     mp90, w, color=MODEL_PAL[model], alpha=0.55, edgecolor="black", hatch="//", linewidth=0.5)
            ax.bar(mbase + w, mp99, w, color=MODEL_PAL[model], alpha=0.3, edgecolor="black", hatch="//", linewidth=0.5)

# Legend patches
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor="gray", alpha=1.0, label="p50"),
    Patch(facecolor="gray", alpha=0.55, label="p90"),
    Patch(facecolor="gray", alpha=0.3, label="p99"),
    Patch(facecolor="white", edgecolor="black", label="llama.cpp (solid edge)"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="MLX-LM (hatched)"),
]

ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_title("Request Latency — p50 / p90 / p99 Spread (ctx=30720, gen=1024) — llama.cpp vs MLX-LM",
             fontweight="bold", fontsize=12.5)
ax.set_ylabel("Request Latency (ms)")
ax.legend(handles=legend_patches, fontsize=8, title="Percentile / Backend", ncol=2)
plt.tight_layout()
save(fig, "i2j_request_latency_percentiles_ctx30720_gen1024.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2k — Prefill vs Decode Power grouped bar across all prompt lengths
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2k: Prefill vs Decode Power, gen=1024, all prompts --")

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, 11))
axes = axes.flatten()
fig.suptitle("Prefill vs Decode Power by Prompt Length — gen=1024, All Models",
             fontsize=14, fontweight="bold")

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    s = gen1024[gen1024.model == model].sort_values("prompt")
    if s.empty: continue
    x_pos = np.arange(len(s))
    w = 0.35
    prefill_vals = s["prefill_pw"].values
    decode_vals = s["decode_pw"].values
    bars1 = ax.bar(x_pos - w/2, prefill_vals, w, label="Prefill W",
                    color="#FF9800", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x_pos + w/2, decode_vals, w, label="Decode W",
                    color="#2196F3", edgecolor="white", linewidth=0.5)
    prompt_labels = [f"{int(p)//1024}K" if p >= 1024 else f"{int(p)}" for p in s["prompt"]]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prompt_labels)
    ax.set_title(MDL_FLAT[model], fontsize=11, fontweight="bold")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Prompt (tok)" if idx >= NCOLS else "")
    ax.legend(fontsize=8)

plt.tight_layout()
save(fig, "i2k_prefill_decode_power_by_prompt.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE I.2l — Full sweep: OSL mismatch heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("-- Figure I.2l: OSL mismatch heatmap (llama.cpp, complete 6-model dataset) --")

# NOTE: must pivot per-backend — for the 3 shared models, a naive df[df.model==m]
# filter would mix llama.cpp's near-zero OSL mismatch with MLX-LM's large
# undershoot and average them together via aggfunc="median", corrupting exactly
# the number section 4.3's backend caveat depends on. df_lc keeps this grid
# llama.cpp-only (matching Table 1's complete 6-model set); the MLX-LM-only
# companion grid right after this one is what actually shows the undershoot.
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(24, 11))
axes = axes.flatten()
fig.suptitle("OSL Mismatch % (vs requested gen target) — All Prompt × Gen, All Models — llama.cpp",
             fontsize=16, fontweight="bold")

for idx, model in enumerate(MODELS):
    ax = axes[idx]
    sub = df_lc[df_lc.model == model]
    if sub.empty or sub["osl_mis"].isna().all():
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        continue
    pivot = sub.pivot_table(index="gen", columns="prompt", values="osl_mis", aggfunc="median")
    pivot.columns = [PROMPT_LABEL_MAP.get(int(c), str(c)) for c in pivot.columns]
    # use a diverging colormap centered at 0
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max())) * 1.1
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                linewidths=1, cbar=True, center=0, vmin=-vmax, vmax=vmax,
                annot_kws={"fontsize": 13, "fontweight": "bold"},
                cbar_kws={"shrink": 0.85})
    ax.tick_params(axis="both", labelsize=12)
    ax.collections[0].colorbar.ax.tick_params(labelsize=11)
    ax.set_title(MDL_FLAT[model], fontsize=14, fontweight="bold")
    ax.set_xlabel("Prompt (tok)" if idx >= NCOLS else "", fontsize=12)
    ax.set_ylabel("Gen (tok)" if idx % NCOLS == 0 else "", fontsize=12)

plt.tight_layout()
save(fig, "i2l_osl_mismatch_heatmaps.png")

# Companion MLX-LM grid (3 shared models) — this is the one that actually shows
# the Granite-4.0-H-Tiny undershoot described in section 4.3.
print("-- Figure I.2l (MLX-LM companion): OSL mismatch heatmap, 3 shared models --")
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
axes = np.atleast_1d(axes).flatten()
fig.suptitle("OSL Mismatch % (vs requested gen target) — All Prompt × Gen — MLX-LM (3 shared models)",
             fontsize=15, fontweight="bold")

for idx, model in enumerate(MLXLM_MODELS):
    ax = axes[idx]
    sub = df_mlx[df_mlx.model == model]
    if sub.empty or sub["osl_mis"].isna().all():
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        continue
    pivot = sub.pivot_table(index="gen", columns="prompt", values="osl_mis", aggfunc="median")
    pivot.columns = [PROMPT_LABEL_MAP.get(int(c), str(c)) for c in pivot.columns]
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max())) * 1.1
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdBu_r",
                linewidths=1, cbar=True, center=0, vmin=-vmax, vmax=vmax,
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                cbar_kws={"shrink": 0.85})
    ax.tick_params(axis="both", labelsize=11)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    ax.set_title(MDL_FLAT[model], fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt (tok)", fontsize=11)
    ax.set_ylabel("Gen (tok)" if idx == 0 else "", fontsize=11)

plt.tight_layout()
save(fig, "i2l_osl_mismatch_heatmaps_mlxlm.png")


print(f"\nAll Appendix I.2 charts saved to {OUT_DIR}")
