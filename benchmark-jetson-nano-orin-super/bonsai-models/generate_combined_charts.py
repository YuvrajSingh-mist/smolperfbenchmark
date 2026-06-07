#!/usr/bin/env python3
"""Generate 4-power-mode comparison charts for the Jetson Orin Nano Super Bonsai benchmark.

Models: Bonsai-1.7B, Bonsai-4B, Bonsai-8B, Ternary-Bonsai-1.7B, Ternary-Bonsai-4B
Sweep: prompt ∈ {256, 512, 1024, 2048} tok × gen ∈ {128, 256, 512} tok × 10 reqs/combo
Canonical standard: ctx=2048, gen=256
Charts saved to bonsai-models/artifacts/charts/ (used by benchmark_report.md)
"""

import json, re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

BONSAI_ROOT = Path(__file__).parent
ARTIFACTS   = BONSAI_ROOT / "artifacts/llamacpp"

# Charts go into the report's artifact directory
OUT_DIR = BONSAI_ROOT / "artifacts/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "7W":   "bonsai-all-20260528-0328-7W",
    "15W":  "bonsai-all-20260525-0243-15W",
    "25W":  "bonsai-all-20260527-0200-25W",
    "MAXN": "bonsai-all-20260526-0239-MAXN_SUPER",
}

MODELS = [
    "Bonsai-1.7B",
    "Bonsai-4B",
    "Bonsai-8B",
    "Ternary-Bonsai-1.7B",
    "Ternary-Bonsai-4B",
]
MODEL_DISPLAY = {
    "Bonsai-1.7B":         "Bonsai\n1.7B",
    "Bonsai-4B":           "Bonsai\n4B",
    "Bonsai-8B":           "Bonsai\n8B",
    "Ternary-Bonsai-1.7B": "Ternary\n1.7B",
    "Ternary-Bonsai-4B":   "Ternary\n4B",
}
MDL = {k: v.replace("\n", " ") for k, v in MODEL_DISPLAY.items()}

PROMPT_LENGTHS = [256, 512, 1024, 2048]
GEN_LENGTHS    = [128, 256, 512]

CANONICAL_CTX = 2048
CANONICAL_GEN = 512

# 2 rows × 3 cols facet layout for 5 models
NROWS, NCOLS = 2, 3

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
WIDTH = 0.18

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

                tok_s    = pct("output_token_throughput_per_user", "p50")
                ttft     = pct("time_to_first_token",  "p50")
                ttft_p90 = pct("time_to_first_token",  "p90")
                ttft_p99 = pct("time_to_first_token",  "p99")
                itl      = pct("inter_token_latency",  "p50")
                itl_p99  = pct("inter_token_latency",  "p99")
                prefill  = pct("prefill_throughput_per_user", "p50")
                rl_p50   = pct("request_latency", "p50")
                e2e      = pct("e2e_output_token_throughput", "p50")
                isl      = pct("input_sequence_length",  "p50")
                osl      = pct("output_sequence_length", "p50")

                t0_str = d.get("start_time")
                t1_str = d.get("end_time")
                if not t0_str or not t1_str:
                    continue
                t0 = datetime.fromisoformat(t0_str).timestamp()
                t1 = datetime.fromisoformat(t1_str).timestamp()

                samp_records = [(ep, mw, tj) for (ep, mw, tj) in tegra if t0 <= ep <= t1]
                if not samp_records:
                    continue
                samps = [(mw, tj) for (_, mw, tj) in samp_records]
                power_w = float(np.median([mw for mw, _ in samps])) / 1000  # p50 power over run window

                # p50 TTFT from aiperf.json (correct: aiperf computes per-request then takes p50)
                p50_ttft_s   = ttft / 1000.0 if ttft else None
                # p50_decode_s computed from per-request ns timestamps after jsonl is read below
                p50_decode_s = None

                # ── Per-request phase power from profile_export.jsonl ──────────
                jsonl_path = p.parent / "profile_export.jsonl"
                prefill_mw, decode_mw = [], []

                if jsonl_path.exists():
                    per_req = []
                    with open(jsonl_path) as fj:
                        for line in fj:
                            line = line.strip()
                            if not line: continue
                            try:
                                rec  = json.loads(line)
                                meta = rec.get("metadata", {})
                                if (meta.get("benchmark_phase") == "profiling"
                                        and "request_start_ns" in meta
                                        and "request_ack_ns"   in meta
                                        and "request_end_ns"   in meta):
                                    per_req.append((meta["request_start_ns"],
                                                    meta["request_ack_ns"],
                                                    meta["request_end_ns"]))
                            except Exception:
                                continue

                    if per_req:
                        # ns timestamps are absolute Unix epoch — divide by 1e9 to get epoch seconds
                        # directly comparable to tegrastats ep without any anchor conversion
                        prefill_wins = [(s / 1e9, a / 1e9) for s, a, e in per_req]
                        decode_wins  = [(a / 1e9, e / 1e9) for s, a, e in per_req]

                        for ep, mw, _ in samp_records:
                            if any(ws <= ep <= wa for ws, wa in prefill_wins):
                                prefill_mw.append(mw)
                            elif any(wa < ep <= we for wa, we in decode_wins):
                                decode_mw.append(mw)

                        # p50 of per-request decode durations — more precise than p50(RL) - p50(TTFT)
                        p50_decode_s = float(np.median([(e - a) / 1e9 for s, a, e in per_req]))

                # fallback to reconstruction if jsonl missing or no samples assigned
                if not prefill_mw or not decode_mw:
                    ttft_avg_ms = pct("time_to_first_token", "p50")
                    rl_p50_ms   = pct("request_latency",     "p50")
                    n_reqs_int  = int(pct("request_count", "avg") or 20)
                    rl_s_fb     = (rl_p50_ms   or 0) / 1000.0
                    ttft_s_fb   = (ttft_avg_ms or 0) / 1000.0
                    prefill_mw, decode_mw = [], []
                    if rl_s_fb > 0 and ttft_s_fb > 0:
                        for ep, mw, _ in samp_records:
                            elapsed       = ep - t0
                            req_idx       = int(elapsed / rl_s_fb)
                            if req_idx >= n_reqs_int: continue
                            phase_elapsed = elapsed - req_idx * rl_s_fb
                            if phase_elapsed <= ttft_s_fb:
                                prefill_mw.append(mw)
                            else:
                                decode_mw.append(mw)
                    if not p50_ttft_s:
                        p50_ttft_s   = ttft_s_fb
                        p50_decode_s = rl_s_fb - ttft_s_fb

                prefill_power_w = float(np.median(prefill_mw)) / 1000 if prefill_mw else power_w
                decode_power_w  = float(np.median(decode_mw))  / 1000 if decode_mw  else power_w

                # ── Energy: exact phase power × p50 phase duration ───────────────
                rl_s   = (rl_p50 or 0) / 1000.0   # p50 kept for latency charts
                ttft_s = (ttft   or 0) / 1000.0   # p50 kept for latency charts

                prefill_j = prefill_power_w * p50_ttft_s   if (prefill_power_w and p50_ttft_s)   else None
                decode_j  = decode_power_w  * p50_decode_s if (decode_power_w  and p50_decode_s) else None
                total_j   = (prefill_j + decode_j)          if (prefill_j and decode_j)            else None

                tok_j        = osl / decode_j  if (osl and decode_j  and decode_j  > 0) else None
                decode_tokj  = tok_j
                prefill_tokj = isl / prefill_j if (isl and prefill_j and prefill_j > 0) else None
                total_tokj   = (isl + osl) / total_j if (
                    isl is not None and osl is not None and total_j and total_j > 0) else None

                rows.append(dict(
                    mode=mode, model=model,
                    prompt=ctx, gen=gen,
                    tok_s=tok_s, ttft=ttft,
                    ttft_p90=ttft_p90, ttft_p99=ttft_p99,
                    itl=itl, itl_p99=itl_p99,
                    prefill=prefill,
                    rl_p50=rl_p50, e2e=e2e,
                    power_w=power_w,
                    prefill_power_w=prefill_power_w, decode_power_w=decode_power_w,
                    tok_j=tok_j,
                    isl=isl, osl=osl,
                    prefill_j=prefill_j, decode_j=decode_j, total_j=total_j,
                    prefill_tokj=prefill_tokj, decode_tokj=decode_tokj, total_tokj=total_tokj,
                ))

df = pd.DataFrame(rows)
modes_avail = [m for m in ["7W", "15W", "25W", "MAXN"] if m in df["mode"].unique()]
n = len(modes_avail)
offsets = np.linspace(-(n - 1) * WIDTH / 2, (n - 1) * WIDTH / 2, n)
print(f"Loaded {len(df)} rows across {df['mode'].nunique()} modes, {df['model'].nunique()} models")
print(f"Models: {df['model'].unique().tolist()}")
print(f"Modes:  {df['mode'].unique().tolist()}")


# ── Print key data tables ──────────────────────────────────────────────────────
print(f"\n=== CANONICAL CELL (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) ===")
can = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
for model in MODELS:
    for mode in ["7W", "15W", "25W", "MAXN"]:
        r = can[(can.model == model) & (can["mode"] == mode)]
        if r.empty:
            print(f"  {model:22s} {mode:4s}  NO DATA")
            continue
        row = r.iloc[0]
        print(f"  {model:22s} {mode:4s}  "
              f"tok_s={row.tok_s:6.1f}  tok_j={row.tok_j:5.2f}  "
              f"ttft={row.ttft:8.1f}ms  itl={row.itl:6.2f}ms  "
              f"power={row.power_w:5.2f}W  rl={row.rl_p50:8.1f}ms")

print("\n=== AVERAGE POWER (W) per model per mode (avg over all combos) ===")
avg_pw = df.groupby(["model", "mode"])["power_w"].median().unstack("mode").reindex(MODELS)
print(avg_pw.round(2).to_string())

print("\n=== THROUGHPUT SPEEDUP RATIOS (mean tok_s over all 12 combos) ===")
toks_mean = df.groupby(["model", "mode"])["tok_s"].median().unstack("mode").reindex(MODELS)
for model in MODELS:
    r = toks_mean.loc[model]
    def sp(a, b):
        if pd.isna(r.get(a)) or pd.isna(r.get(b)) or r.get(b) == 0: return float('nan')
        return r[a] / r[b]
    print(f"  {model:22s}  25W/15W={sp('25W','15W'):.2f}x  MAXN/15W={sp('MAXN','15W'):.2f}x  "
          f"15W/7W={sp('15W','7W'):.2f}x  25W/7W={sp('25W','7W'):.2f}x  "
          f"MAXN/7W={sp('MAXN','7W'):.2f}x  MAXN/25W={sp('MAXN','25W'):.2f}x")

print("\n=== BEST TOTAL TOK/J per model (max over all mode×ctx×gen) ===")
best_total = df.groupby(["model", "mode", "prompt", "gen"])["total_tokj"].median()
for model in MODELS:
    try:
        sub = best_total[model]
        idx = sub.idxmax()
        print(f"  {model:22s}  best={sub.max():.1f}  at {idx}")
    except Exception:
        print(f"  {model:22s}  NO DATA")


def mode_legend_handles():
    return [mlines.Line2D([], [], color=MODE_PAL[m], ls=ls, marker=mk, lw=2, label=m, ms=7)
            for m, (ls, mk) in MODE_STYLE.items() if m in modes_avail]

def model_legend_handles(models):
    return [mlines.Line2D([], [], color=MODEL_PAL[m], ls="-", lw=3, label=MDL[m])
            for m in models if m in MODEL_PAL]


# ── Helper: 2×3 facet line chart ──────────────────────────────────────────────
def faceted_line_chart(df_in, y_field, y_label, gen_val, fname,
                       y_scale=None, x_is_prompt=True):
    x_vals  = PROMPT_LENGTHS if x_is_prompt else GEN_LENGTHS
    x_label = "Prompt (tok)" if x_is_prompt else "Gen (tok)"
    data = df_in[(df_in.gen == gen_val) & (df_in["mode"].isin(modes_avail))][y_field].dropna()
    if data.empty:
        return
    y_max = y_scale if y_scale else data.max() * 1.15
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 8), sharey=True)
    axes = axes.flatten()
    fig.suptitle(
        f"{y_label} vs {'Prompt' if x_is_prompt else 'Gen'} Length — All Power Modes (gen={gen_val} tok)",
        fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        sub = df_in[(df_in.model == model) & (df_in.gen == gen_val) &
                    (df_in["mode"].isin(modes_avail))]
        for mode, (ls, mk) in MODE_STYLE.items():
            s = sub[sub["mode"] == mode].sort_values("prompt" if x_is_prompt else "gen")
            if s.empty:
                continue
            xvals = s.prompt if x_is_prompt else s.gen
            ax.plot(xvals, s[y_field], marker=mk, ls=ls, lw=2,
                    color=MODE_PAL[mode], label=mode, ms=7)
        ax.set_title(MDL[model], fontsize=10, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label if idx % NCOLS == 0 else "", fontsize=8)
        ax.set_xticks(x_vals)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=7, loc="best")
    # Hide unused panels
    for idx in range(len(MODELS), NROWS * NCOLS):
        axes[idx].set_visible(False)
    plt.tight_layout()
    save(fig, fname)


def facet_vs_gen(ctx_val, y_field, y_label, fname):
    data_sub = df[(df.prompt == ctx_val) & (df["mode"].isin(modes_avail))][y_field].dropna()
    if data_sub.empty:
        return
    y_max = data_sub.max() * 1.15
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 8), sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"{y_label} vs Gen Length — All Power Modes (ctx={ctx_val})",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        sub = df[(df.model == model) & (df.prompt == ctx_val) &
                 (df["mode"].isin(modes_avail))]
        for mode, (ls, mk) in MODE_STYLE.items():
            s = sub[sub["mode"] == mode].sort_values("gen")
            if s.empty:
                continue
            ax.plot(s.gen, s[y_field], marker=mk, ls=ls, lw=2,
                    color=MODE_PAL[mode], label=mode, ms=7)
        ax.set_title(MDL[model], fontsize=10, fontweight="bold")
        ax.set_xlabel("Gen (tok)", fontsize=8)
        ax.set_ylabel(y_label if idx % NCOLS == 0 else "", fontsize=8)
        ax.set_xticks(GEN_LENGTHS)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=7, loc="best")
    for idx in range(len(MODELS), NROWS * NCOLS):
        axes[idx].set_visible(False)
    plt.tight_layout()
    save(fig, fname)


def canonical_bar(ctx_val, gen_val, fname_prefix, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(MODELS))
    for i, mode in enumerate(modes_avail):
        toks_vals, tokj_vals = [], []
        for m in MODELS:
            sub = df[(df.model == m) & (df["mode"] == mode) &
                     (df.prompt == ctx_val) & (df.gen == gen_val)]
            toks_vals.append(sub["tok_s"].median() if not sub.empty else np.nan)
            tokj_vals.append(sub["tok_j"].median() if not sub.empty else np.nan)
        bars1 = ax1.bar(x + offsets[i], toks_vals, WIDTH, label=mode,
                        color=MODE_PAL[mode], edgecolor="white")
        bars2 = ax2.bar(x + offsets[i], tokj_vals, WIDTH, label=mode,
                        color=MODE_PAL[mode], edgecolor="white")
        for bar, v in zip(bars1, toks_vals):
            if not np.isnan(v):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
        for bar, v in zip(bars2, tokj_vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=9)
    ax1.set_title(f"Output Tok/s — ctx={ctx_val}, gen={gen_val}", fontweight="bold")
    ax1.set_ylabel("Output tokens per second")
    ax2.set_title(f"Output Tok/J — ctx={ctx_val}, gen={gen_val}", fontweight="bold")
    ax2.set_ylabel("Output tokens per joule")
    fig.suptitle(
        f"Bonsai Models — All 4 Power Modes: ctx={ctx_val} tok prompt, gen={gen_val} tok output{title_suffix}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, f"{fname_prefix}_ctx{ctx_val}_gen{gen_val}.png")


# ════════════════════════════════════════════════════════════════════════════════
print(f"\n── Main charts (canonical ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) ──")

# 1. Output Tok/s vs Prompt: 2×3 facet, canonical gen
faceted_line_chart(df, "tok_s", "Output Tok/s", CANONICAL_GEN, f"1_tok_s_vs_prompt_gen{CANONICAL_GEN}.png")

# 2. Output Tok/J vs Prompt: 2×3 facet, canonical gen
faceted_line_chart(df, "tok_j", "Output Tok/J", CANONICAL_GEN, f"2_tok_j_vs_prompt_gen{CANONICAL_GEN}.png")

# 3. Best Tok/J grouped bar
best = df.groupby(["model", "mode"])["tok_j"].max().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = [best.loc[m, mode] if (m in best.index and mode in best.columns
                                   and not pd.isna(best.loc[m, mode])) else 0
            for m in MODELS]
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Best Output Tok/J per Model — All Power Modes", fontweight="bold", fontsize=12)
ax.set_ylabel("Output Tok/J")
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "3_best_tok_j_bar.png")

# 4. Average power grouped bar
avg_pw = df.groupby(["model", "mode"])["power_w"].median().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = [avg_pw.loc[m, mode] if (m in avg_pw.index and mode in avg_pw.columns
                                     and not pd.isna(avg_pw.loc[m, mode])) else 0
            for m in MODELS]
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}W", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Average Power Draw per Model — All Power Modes (VDD_CPU_GPU_CV)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Power (W)")
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "4_avg_power_bar.png")

# 5. TTFT p50 grouped bar — canonical ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(12, 6))
x_bar = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["ttft"].median() if not sub.empty else np.nan)
    bars = ax.bar(x_bar + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x_bar)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"TTFT p50 by Power Mode (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("TTFT p50 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "5_ttft_vs_prompt.png")

# 6. Speedup vs 15W baseline
toks_by_mode = df.groupby(["model", "mode"])["tok_s"].median().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(12, 6))
for i, cmp_mode in enumerate(["7W", "25W", "MAXN"]):
    if cmp_mode not in toks_by_mode.columns:
        continue
    speedups = []
    for m in MODELS:
        base = toks_by_mode.loc[m, "15W"] if "15W" in toks_by_mode.columns else None
        cmp  = toks_by_mode.loc[m, cmp_mode] if cmp_mode in toks_by_mode.columns else None
        if base and cmp and not pd.isna(base) and not pd.isna(cmp):
            speedups.append(cmp / base)
        else:
            speedups.append(np.nan)
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
ax.set_title("Output Throughput Speedup vs 15W — avg over all prompt × gen combos",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Tok/s ratio (vs 15W)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "6_speedup_vs_15w.png")

# 7a. Tok/J heatmap — Standard Bonsai models (1.7B, 4B, 8B) at all 4 modes
models_std = ["Bonsai-1.7B", "Bonsai-4B", "Bonsai-8B"]
all_tokj = df["tok_j"].dropna()
vmin, vmax = all_tokj.min(), all_tokj.max()

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
for row, mode in enumerate(["7W", "15W", "25W", "MAXN"]):
    for col, model in enumerate(models_std):
        ax = axes[row][col]
        sub = df[(df.model == model) & (df["mode"] == mode)]
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_visible(False)
            continue
        pivot = sub.pivot_table(index="gen", columns="prompt", values="tok_j", aggfunc="median")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="YlGnBu",
                    linewidths=0.5, cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"{MDL[model]} ({mode})", fontsize=9)
        ax.set_xlabel("Prompt (tok)" if row == 3 else "")
        ax.set_ylabel("Gen (tok)" if col == 0 else "")
fig.suptitle("Output Tok/J Heatmap — Standard Bonsai models at all 4 power modes\n"
             "(rows = gen length, cols = prompt length)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "7a_tok_j_heatmap_small_models.png")

# 7b. Tok/J heatmap — Ternary Bonsai models at all modes
models_tern = ["Ternary-Bonsai-1.7B", "Ternary-Bonsai-4B"]
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
for row, mode in enumerate(["7W", "15W", "25W", "MAXN"]):
    for col, model in enumerate(models_tern):
        ax = axes[row][col]
        sub = df[(df.model == model) & (df["mode"] == mode)]
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        pivot = sub.pivot_table(index="gen", columns="prompt", values="tok_j", aggfunc="median")
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="YlGnBu",
                    linewidths=0.5, cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"{MDL[model]} ({mode})", fontsize=9)
        ax.set_xlabel("Prompt (tok)" if row == 3 else "")
        ax.set_ylabel("Gen (tok)" if col == 0 else "")
fig.suptitle("Output Tok/J Heatmap — Ternary Bonsai models at all 4 power modes\n"
             "(N/A = model did not run at that mode)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "7b_tok_j_heatmap_large_models.png")

# 8. ITL comparison: canonical ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["itl"].median() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Inter-Token Latency p50 — All Power Modes (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("ITL p50 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "8_itl_compare.png")

# 8b. ITL p50 heatmap: models × power modes
itl_matrix = []
for m in MODELS:
    row_vals = []
    for mode in ["7W", "15W", "25W", "MAXN"]:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        row_vals.append(sub["itl"].median() if not sub.empty else np.nan)
    itl_matrix.append(row_vals)
itl_df = pd.DataFrame(itl_matrix,
                       index=[MDL[m] for m in MODELS],
                       columns=["7W", "15W", "25W", "MAXN"])
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(itl_df, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "ITL p50 (ms)"}, ax=ax)
ax.set_title(f"ITL p50 (ms) — All Models × All Power Modes\n(ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Power Mode")
ax.set_ylabel("Model")
plt.tight_layout()
save(fig, "8b_itl_heatmap_models_modes.png")

# 8c. Phase power heatmap: prefill_power_w and decode_power_w — models × modes (canonical cell)
for phase_col, phase_label, cmap in [
    ("prefill_power_w", "Prefill Power (W)", "OrRd"),
    ("decode_power_w",  "Decode Power (W)",  "Blues"),
]:
    matrix = []
    for m in MODELS:
        row_vals = []
        for mode in ["7W", "15W", "25W", "MAXN"]:
            sub = df[(df.model == m) & (df["mode"] == mode) &
                     (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
            row_vals.append(sub[phase_col].median() if not sub.empty else np.nan)
        matrix.append(row_vals)
    phase_df = pd.DataFrame(matrix,
                            index=[MDL[m] for m in MODELS],
                            columns=["7W", "15W", "25W", "MAXN"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(phase_df, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": phase_label}, ax=ax)
    ax.set_title(f"{phase_label} — All Models × All Power Modes\n(ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Power Mode")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fname = "EP_prefill_power_heatmap_canonical.png" if "prefill" in phase_col else "EP_decode_power_heatmap_canonical.png"
    save(fig, fname)

# 9. Request latency (E2E) comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["rl_p50"].median() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Request Latency (E2E) p50 — All Power Modes (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Request latency p50 (ms)")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "10_request_latency_compare.png")

# 10. Prefill throughput: canonical gen, avg over all prompts
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["prefill"].median() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Prefill Throughput — All Power Modes (gen={CANONICAL_GEN}, avg over all prompts)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Prefill tok/s")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "9_prefill_compare.png")

# 11. Canonical cell comparison: ctx=2048, gen=256
canonical_bar(CANONICAL_CTX, CANONICAL_GEN,
              fname_prefix="11_canonical_cell_comparison",
              title_suffix="  [canonical standard]")

# 12. Bonsai-8B spotlight — tok/J at all 4 power modes across gen lengths
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
model = "Bonsai-8B"
for ax, gen_val in zip(axes, GEN_LENGTHS):
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
fig.suptitle("Bonsai-8B: Output Tok/J at All 4 Power Modes across gen lengths",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "12_bonsai_8b_tok_j_spotlight.png")

# ── Energy charts ──────────────────────────────────────────────────────────────
# E_total_energy_vs_gen_length: total_J vs gen at 25W, ctx=2048
fig, ax = plt.subplots(figsize=(10, 6))
sub25 = df[(df["mode"] == "25W") & (df.prompt == CANONICAL_CTX)]
for model in MODELS:
    s = sub25[sub25.model == model].sort_values("gen")
    if s.empty:
        continue
    ax.plot(s.gen, s.total_j, marker="s", ls="--", lw=2,
            color=MODEL_PAL[model], label=MDL[model], ms=8)
ax.set_title("Total Energy per Request vs Gen Length (25W, ctx=2048)",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Gen length (tok)")
ax.set_ylabel("Total energy (J)")
ax.set_xticks(GEN_LENGTHS)
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "E_total_energy_vs_gen_length.png")

# E_mj_per_output_token: mJ/output_tok at ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        if sub.empty or sub["decode_tokj"].isna().all():
            vals.append(np.nan)
        else:
            dtj = sub["decode_tokj"].median()
            vals.append(1000.0 / dtj if dtj and dtj > 0 else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([MDL[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Decode Energy per Output Token in mJ (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("mJ per output token")
ax.legend(fontsize=9)
plt.tight_layout()
save(fig, "E_mj_per_output_token.png")

# ── ITL heatmaps per power mode (EH) ──────────────────────────────────────────
for mode_name in modes_avail:
    sub = df[df["mode"] == mode_name]
    if sub.empty:
        continue
    pivot = sub.pivot_table(index="gen", columns="prompt", values="itl", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="Oranges",
                linewidths=0.5, cbar=True)
    ax.set_title(f"ITL p50 (ms) heatmap: {mode_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Gen (tok)")
    plt.tight_layout()
    save(fig, f"EH_itl_heatmap_{mode_name.lower()}.png")

# ── TTFT heatmaps per power mode (EG) ─────────────────────────────────────────
for mode_name in modes_avail:
    sub = df[df["mode"] == mode_name]
    if sub.empty:
        continue
    pivot = sub.pivot_table(index="gen", columns="prompt", values="ttft", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="Reds",
                linewidths=0.5, cbar=True)
    ax.set_title(f"TTFT p50 (ms) heatmap: {mode_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Gen (tok)")
    plt.tight_layout()
    save(fig, f"EG_ttft_heatmap_{mode_name.lower()}.png")

# ── Prefill throughput heatmaps per power mode (EI) ───────────────────────────
for mode_name in modes_avail:
    sub = df[df["mode"] == mode_name]
    if sub.empty:
        continue
    pivot = sub.pivot_table(index="gen", columns="prompt", values="prefill", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="Greens",
                linewidths=0.5, cbar=True)
    ax.set_title(f"Prefill tok/s heatmap: {mode_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel("Gen (tok)")
    plt.tight_layout()
    save(fig, f"EI_prefill_tput_heatmap_{mode_name.lower()}.png")

# ── Appendix E: tok/s, tok/J, and phase-power heatmaps at each power mode ──────
print("\n── Appendix E charts ──")
for metric, label, fmt_str, cmap in [
    ("tok_s",           "Output Tok/s",      ".1f", "Blues"),
    ("tok_j",           "Output Tok/J",      ".2f", "YlGnBu"),
    ("prefill_power_w", "Prefill Power (W)", ".2f", "OrRd"),
    ("decode_power_w",  "Decode Power (W)",  ".2f", "PuBu"),
]:
    for mode_name in modes_avail:
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 10))
        axes = axes.flatten()
        fig.suptitle(f"{label} — All Prompt × Gen Combinations at {mode_name}",
                     fontsize=14, fontweight="bold")
        for idx, model in enumerate(MODELS):
            ax = axes[idx]
            sub = df[(df.model == model) & (df["mode"] == mode_name)]
            if sub.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            pivot = sub.pivot_table(index="gen", columns="prompt", values=metric, aggfunc="median")
            sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt_str, cmap=cmap,
                        linewidths=0.5, cbar=False)
            ax.set_title(MDL[model], fontsize=10, fontweight="bold")
            ax.set_xlabel("Prompt (tok)" if idx >= NCOLS else "")
            ax.set_ylabel("Gen (tok)" if idx % NCOLS == 0 else "")
        for idx in range(len(MODELS), NROWS * NCOLS):
            axes[idx].set_visible(False)
        plt.tight_layout()
        short = metric.replace("_power_w", "_power").replace("output_token_", "")
        save(fig, f"E_{short}_heatmap_{mode_name.lower()}.png")

# ── Comparison line charts (all metrics × gen/prompt combos) ──────────────────
print("\n── Comparison line charts ──")

for gen_val in GEN_LENGTHS:
    prefix = "EF_req_latency" if gen_val < 256 else "22a_request_latency"
    faceted_line_chart(df, "rl_p50", "Request latency (ms)", gen_val,
                       f"{prefix}_vs_prompt_gen{gen_val}.png")

for gen_val in GEN_LENGTHS:
    for field, label, prefix in [
        ("prefill_tokj", "Prefill tok/J", "22e"),
        ("total_tokj",   "Total tok/J",   "22g"),
    ]:
        faceted_line_chart(df, field, label, gen_val,
                           f"{prefix}_{field}_vs_prompt_gen{gen_val}.png")
    if gen_val == CANONICAL_GEN:
        faceted_line_chart(df, "decode_tokj", "Decode tok/J", CANONICAL_GEN,
                           f"22f_decode_tokj_vs_prompt_gen{CANONICAL_GEN}.png")

for ctx_val in PROMPT_LENGTHS:
    facet_vs_gen(ctx_val, "decode_tokj", "Decode tok/J",
                 f"EF_decode_tokj_vs_gen_ctx{ctx_val}.png")

for gen_val in [128, 256]:
    for field, label in [("prefill_tokj", "Prefill tok/J"), ("total_tokj", "Total tok/J")]:
        faceted_line_chart(df, field, label, gen_val,
                           f"EF_{field}_vs_prompt_gen{gen_val}.png")

for gen_val in GEN_LENGTHS:
    faceted_line_chart(df, "ttft",    "TTFT p50 (ms)",  gen_val,
                       f"EG_ttft_vs_prompt_gen{gen_val}.png")
    faceted_line_chart(df, "itl",     "ITL p50 (ms)",   gen_val,
                       f"EH_itl_vs_prompt_gen{gen_val}.png")
    faceted_line_chart(df, "prefill", "Prefill tok/s",  gen_val,
                       f"EI_prefill_tput_vs_prompt_gen{gen_val}.png")

for ctx_val in PROMPT_LENGTHS:
    facet_vs_gen(ctx_val, "ttft",    "TTFT p50 (ms)", f"EG_ttft_vs_gen_ctx{ctx_val}.png")
    facet_vs_gen(ctx_val, "itl",     "ITL p50 (ms)",  f"EH_itl_vs_gen_ctx{ctx_val}.png")
    facet_vs_gen(ctx_val, "prefill", "Prefill tok/s", f"EI_prefill_tput_vs_gen_ctx{ctx_val}.png")

faceted_line_chart(df, "rl_p50",    "Request latency (ms)", 256,
                   "EA_request_latency_vs_prompt_all_models.png")
faceted_line_chart(df, "prefill_j", "Prefill energy (J)",   256,
                   "EA_prefill_energy_vs_prompt_all_models.png")

print(f"\nAll charts saved to {OUT_DIR}")
