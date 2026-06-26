#!/usr/bin/env python3
"""Generate 4-power-mode comparison charts for the Jetson Orin Nano Super benchmark.

Canonical standard: ctx=2048, gen=256 (highest sweep point).
Also generates all 12 prompt×gen combination charts for Appendix E.

Power / tok/J method (ported from bonsai generate_combined_charts.py):
  - Reads tegrastats.log at the artifact root (500ms samples, VDD_CPU_GPU_CV rail)
  - Uses profile_export.jsonl per-request ns timestamps to split samples into
    prefill (request_start_ns → request_ack_ns) and decode (ack_ns → end_ns) windows
  - decode_power_w = median of samples falling inside decode windows
  - p50_decode_s   = median of per-request decode durations from jsonl
  - tok_j          = OSL / (decode_power_w × p50_decode_s)   [output tokens per joule]
  - Falls back to timeline reconstruction if profile_export.jsonl is absent or
    yields no samples (tegrastats interval may not land in short phases).
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

# ── Run registry ──────────────────────────────────────────────────────────────
# Each entry: mode_label → (artifact_dir_name, backend_subdir)
# backend_subdir = "" means model dirs are directly under the artifact root
# (HF-downloaded llamacpp data; local ollama data sits under "ollama/")
RUNS = {
    # llama.cpp / CUDA (HF datasets)
    "7W":        ("llamacpp-hf-7w",             ""),
    "15W":       ("llamacpp-hf-15w",            ""),
    "25W":       ("llamacpp-hf-25w",            ""),
    "MAXN":      ("llamacpp-hf-maxn",           ""),
    # Ollama (local, have tegrastats.log)
    "7W-ollama":   ("blog-all-20260607-0403-7w",   "ollama"),
    "15W-ollama":  ("blog-all-20260606-0139-15w",  "ollama"),
    "25W-ollama":  ("blog-all-20260622-0159-25w",  "ollama"),
    "MAXN-ollama": ("blog-all-20260621-1401-maxn", "ollama"),
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

NROWS, NCOLS = 2, 4   # 2×4 for 8 models

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
BACKEND_GROUP = {
    "7W": "llamacpp", "15W": "llamacpp", "25W": "llamacpp", "MAXN": "llamacpp",
    "7W-ollama": "ollama", "15W-ollama": "ollama", "25W-ollama": "ollama", "MAXN-ollama": "ollama",
}
POWER_LEVEL = {
    "7W": "7W", "7W-ollama": "7W",
    "15W": "15W", "15W-ollama": "15W",
    "25W": "25W", "25W-ollama": "25W",
    "MAXN": "MAXN", "MAXN-ollama": "MAXN",
}
# Ollama lines/bars appear at 65% opacity so llamacpp stays visually primary;
# all ollama use "--" dashed style to distinguish backend at a glance.
BACKEND_ALPHA = {
    "7W": 1.0, "15W": 1.0, "25W": 1.0, "MAXN": 1.0,
    "7W-ollama": 0.65, "15W-ollama": 0.65, "25W-ollama": 0.65, "MAXN-ollama": 0.65,
}
MODEL_PAL = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
WIDTH = 0.10

sns.set_theme(style="darkgrid", font_scale=1.05)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── load tegrastats per run ───────────────────────────────────────────────────
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


# ── build dataframe ───────────────────────────────────────────────────────────
rows = []
for mode, (art, backend) in RUNS.items():
    tegra = get_tegra(art)
    for model in MODELS:
        for gen in GEN_LENGTHS:
            for ctx in PROMPT_LENGTHS:
                if backend:
                    p = ARTIFACTS / art / backend / model / f"gen{gen}" / f"ctx{ctx}" / "profile_export_aiperf.json"
                else:
                    p = ARTIFACTS / art / model / f"gen{gen}" / f"ctx{ctx}" / "profile_export_aiperf.json"
                if not p.exists():
                    continue
                try:
                    d = json.loads(p.read_text())
                except Exception:
                    continue

                def pct(k, v="p50"):
                    return (d.get(k) or {}).get(v)

                tok_s    = pct("output_token_throughput_per_user", "p50")
                ttft     = pct("time_to_first_token",              "p50")
                ttft_p90 = pct("time_to_first_token",              "p90")
                ttft_p99 = pct("time_to_first_token",              "p99")
                itl      = pct("inter_token_latency",              "p50")
                itl_p99  = pct("inter_token_latency",              "p99")
                prefill  = pct("prefill_throughput_per_user",      "p50")
                rl_p50   = pct("request_latency",                  "p50")
                rl_avg   = pct("request_latency",                  "avg")
                e2e      = pct("e2e_output_token_throughput",      "p50")
                isl      = pct("input_sequence_length",            "p50")
                osl      = pct("output_sequence_length",           "p50")

                t0_str = d.get("start_time")
                t1_str = d.get("end_time")
                if not t0_str or not t1_str:
                    continue
                t0 = datetime.fromisoformat(t0_str).timestamp()
                t1 = datetime.fromisoformat(t1_str).timestamp()

                samp_records = [(ep, mw, tj) for (ep, mw, tj) in tegra if t0 <= ep <= t1]
                power_w = (float(np.median([mw for ep, mw, tj in samp_records])) / 1000
                           if samp_records else None)

                p50_ttft_s   = ttft / 1000.0 if ttft else None
                p50_decode_s = None

                # ── Per-request phase power from profile_export.jsonl ─────────
                # Assigns each tegrastats sample to prefill or decode window using
                # exact per-request ns timestamps -- same method as bonsai charts.
                jsonl_path = p.parent / "profile_export.jsonl"
                prefill_mw, decode_mw = [], []

                if jsonl_path.exists() and samp_records:
                    per_req = []
                    with open(jsonl_path) as fj:
                        for line in fj:
                            line = line.strip()
                            if not line:
                                continue
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
                        prefill_wins = [(s / 1e9, a / 1e9) for s, a, e in per_req]
                        decode_wins  = [(a / 1e9, e / 1e9) for s, a, e in per_req]

                        for ep, mw, _ in samp_records:
                            if any(ws <= ep <= wa for ws, wa in prefill_wins):
                                prefill_mw.append(mw)
                            elif any(wa < ep <= we for wa, we in decode_wins):
                                decode_mw.append(mw)

                        p50_decode_s = float(np.median([(e - a) / 1e9 for s, a, e in per_req]))

                # Fallback: timeline reconstruction if jsonl absent or no samples classified
                if (not prefill_mw or not decode_mw) and samp_records:
                    rl_p50_ms  = pct("request_latency", "p50")
                    n_reqs_int = int(pct("request_count", "avg") or 20)
                    rl_s_fb    = (rl_p50_ms or 0) / 1000.0
                    ttft_s_fb  = (ttft      or 0) / 1000.0
                    prefill_mw, decode_mw = [], []
                    if rl_s_fb > 0 and ttft_s_fb > 0:
                        for ep, mw, _ in samp_records:
                            elapsed       = ep - t0
                            req_idx       = int(elapsed / rl_s_fb)
                            if req_idx >= n_reqs_int:
                                continue
                            phase_elapsed = elapsed - req_idx * rl_s_fb
                            if phase_elapsed <= ttft_s_fb:
                                prefill_mw.append(mw)
                            else:
                                decode_mw.append(mw)
                    if p50_decode_s is None:
                        p50_decode_s = rl_s_fb - ttft_s_fb if rl_s_fb > ttft_s_fb else None

                prefill_power_w = float(np.median(prefill_mw)) / 1000 if prefill_mw else power_w
                decode_power_w  = float(np.median(decode_mw))  / 1000 if decode_mw  else power_w

                # ── Energy: exact phase power × p50 phase duration ────────────
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
                    rl_p50=rl_p50, rl_avg=rl_avg, e2e=e2e,
                    power_w=power_w,
                    prefill_power_w=prefill_power_w, decode_power_w=decode_power_w,
                    tok_j=tok_j,
                    isl=isl, osl=osl,
                    prefill_j=prefill_j, decode_j=decode_j, total_j=total_j,
                    prefill_tokj=prefill_tokj, decode_tokj=decode_tokj, total_tokj=total_tokj,
                ))

df = pd.DataFrame(rows)

# 7W llama.cpp has no tegrastats (not retained for that run).
# Approximate tok_j = tok_s / avg_power_W using whole-run power from original local logs.
# These values match LLAMACPP_POWER["7W"] in generate_merged_charts.py and Table 6 in the report.
_LC_7W_POWER = {
    "smollm2-135m": 1.99, "smollm2-360m": 2.27, "qwen2.5-0.5b": 2.22,
    "lfm2.5-350m":  2.10, "lfm2.5-1.2b":  2.34, "qwen3-0.6b":   1.98,
    "llama3.2-1b":  2.26, "gemma3-1b":    1.96,
}
_mask_7w = (df["mode"] == "7W") & df["tok_j"].isna()
for _model, _pwr in _LC_7W_POWER.items():
    _m = _mask_7w & (df["model"] == _model)
    df.loc[_m, "tok_j"]       = df.loc[_m, "tok_s"] / _pwr
    df.loc[_m, "decode_tokj"] = df.loc[_m, "tok_s"] / _pwr
    df.loc[_m, "power_w"]     = _pwr
    # prefill_tokj = ISL / (avg_power * TTFT_s); total_tokj = (ISL+OSL) / (avg_power * RL_s)
    ttft_s = df.loc[_m, "ttft"] / 1000.0
    rl_s   = df.loc[_m, "rl_p50"] / 1000.0
    df.loc[_m, "prefill_tokj"] = df.loc[_m, "isl"] / (_pwr * ttft_s)
    df.loc[_m, "total_tokj"]   = (df.loc[_m, "isl"] + df.loc[_m, "osl"]) / (_pwr * rl_s)

modes_avail = [m for m in ["7W", "15W", "25W", "MAXN",
                            "7W-ollama", "15W-ollama", "25W-ollama", "MAXN-ollama"]
               if m in df["mode"].unique()]
n = len(modes_avail)
offsets = np.linspace(-(n - 1) * WIDTH / 2, (n - 1) * WIDTH / 2, n)
print(f"Loaded {len(df)} rows across {df['mode'].nunique()} modes, {df['model'].nunique()} models")
print(f"Models: {df['model'].unique().tolist()}")
print(f"Modes:  {df['mode'].unique().tolist()}")

# Print key data tables
print(f"\n=== CANONICAL CELL (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) ===")
can = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
for model in MODELS:
    for mode in modes_avail:
        r = can[(can.model == model) & (can["mode"] == mode)]
        if r.empty:
            continue
        row = r.iloc[0]
        tok_j_str = f"{row.tok_j:.3f}" if row.tok_j is not None and not pd.isna(row.tok_j) else "N/A"
        pw_str    = f"{row.power_w:.2f}W" if row.power_w is not None and not pd.isna(row.power_w) else "N/A"
        print(f"  {model:22s} {mode:12s}  tok_s={row.tok_s or 0:6.1f}  tok_j={tok_j_str:>8}  "
              f"ttft={row.ttft or 0:8.1f}ms  itl={row.itl or 0:6.2f}ms  power={pw_str}")

print("\n=== AVERAGE POWER (W) per model per mode (median over all combos) ===")
avg_pw = df.groupby(["model", "mode"])["power_w"].median().unstack("mode").reindex(MODELS)
print(avg_pw.round(2).to_string())

print("\n=== BEST TOK/J per model per mode ===")
best_tokj = df.groupby(["model", "mode"])["tok_j"].max().unstack("mode").reindex(MODELS)
print(best_tokj.round(3).to_string())


def mode_legend_handles():
    return [mlines.Line2D([], [], color=MODE_PAL[m], ls=ls, marker=mk, lw=2, label=m, ms=7,
                          alpha=BACKEND_ALPHA.get(m, 1.0))
            for m, (ls, mk) in MODE_STYLE.items() if m in modes_avail]

def model_legend_handles(models):
    return [mlines.Line2D([], [], color=MODEL_PAL[m], ls="-", lw=3, label=MDL[m])
            for m in models if m in MODEL_PAL]


# ── Helper: 3×3 facet line chart ──────────────────────────────────────────────
def faceted_line_chart(df_in, y_field, y_label, gen_val, fname, y_scale=None, x_is_prompt=True):
    x_vals  = PROMPT_LENGTHS if x_is_prompt else GEN_LENGTHS
    x_label = "Prompt (tok)" if x_is_prompt else "Gen (tok)"
    data = df_in[(df_in.gen == gen_val) & (df_in["mode"].isin(modes_avail))][y_field].dropna()
    if data.empty:
        return
    y_max = y_scale if y_scale else data.max() * 1.15
    active = [m for m in MODELS if not df_in[df_in.model == m].empty]
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 12), sharey=True)
    axes = axes.flatten()
    fig.suptitle(
        f"{y_label} vs {'Prompt' if x_is_prompt else 'Gen'} Length -- All Power Modes (gen={gen_val} tok)",
        fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        if idx >= NROWS * NCOLS:
            break
        ax = axes[idx]
        sub = df_in[(df_in.model == model) & (df_in.gen == gen_val) &
                    (df_in["mode"].isin(modes_avail))]
        for mode, (ls, mk) in MODE_STYLE.items():
            if mode not in modes_avail:
                continue
            s = sub[sub["mode"] == mode].sort_values("prompt" if x_is_prompt else "gen")
            if s.empty:
                continue
            xvals = s.prompt if x_is_prompt else s.gen
            ax.plot(xvals, s[y_field], marker=mk, ls=ls, lw=2,
                    color=MODE_PAL[mode], label=mode, ms=7,
                    alpha=BACKEND_ALPHA.get(mode, 1.0))
        ax.set_title(MDL.get(model, model), fontsize=10, fontweight="bold")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label if idx % NCOLS == 0 else "", fontsize=8)
        ax.set_xticks(x_vals)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis='x', labelsize=7)
    for idx in range(len(MODELS), NROWS * NCOLS):
        axes[idx].set_visible(False)
    fig.legend(handles=mode_legend_handles(), loc="lower center",
               ncol=4, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0),
               markerscale=1.4, handlelength=2.5, handletextpad=0.8,
               columnspacing=1.2)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    save(fig, fname)


def facet_vs_gen(ctx_val, y_field, y_label, fname):
    data_sub = df[(df.prompt == ctx_val) & (df["mode"].isin(modes_avail))][y_field].dropna()
    if data_sub.empty:
        return
    y_max = data_sub.max() * 1.15
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(20, 12), sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"{y_label} vs Gen Length -- All Power Modes (ctx={ctx_val})",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        if idx >= NROWS * NCOLS:
            break
        ax = axes[idx]
        sub = df[(df.model == model) & (df.prompt == ctx_val) &
                 (df["mode"].isin(modes_avail))]
        for mode, (ls, mk) in MODE_STYLE.items():
            if mode not in modes_avail:
                continue
            s = sub[sub["mode"] == mode].sort_values("gen")
            if s.empty:
                continue
            ax.plot(s.gen, s[y_field], marker=mk, ls=ls, lw=2,
                    color=MODE_PAL[mode], label=mode, ms=7,
                    alpha=BACKEND_ALPHA.get(mode, 1.0))
        ax.set_title(MDL.get(model, model), fontsize=10, fontweight="bold")
        ax.set_xlabel("Gen (tok)", fontsize=8)
        ax.set_ylabel(y_label if idx % NCOLS == 0 else "", fontsize=8)
        ax.set_xticks(GEN_LENGTHS)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis='x', labelsize=7)
    for idx in range(len(MODELS), NROWS * NCOLS):
        axes[idx].set_visible(False)
    fig.legend(handles=mode_legend_handles(), loc="lower center",
               ncol=4, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0),
               markerscale=1.4, handlelength=2.5, handletextpad=0.8,
               columnspacing=1.2)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    save(fig, fname)


def canonical_bar(ctx_val, gen_val, fname_prefix, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6.5))
    x = np.arange(len(MODELS))
    for i, mode in enumerate(modes_avail):
        toks_vals, tokj_vals = [], []
        for m in MODELS:
            sub = df[(df.model == m) & (df["mode"] == mode) &
                     (df.prompt == ctx_val) & (df.gen == gen_val)]
            toks_vals.append(sub["tok_s"].median() if not sub.empty else np.nan)
            tokj_vals.append(sub["tok_j"].median() if not sub.empty else np.nan)
        al = BACKEND_ALPHA.get(mode, 1.0)
        bars1 = ax1.bar(x + offsets[i], toks_vals, WIDTH, label=mode,
                        color=MODE_PAL[mode], edgecolor="white", alpha=al)
        bars2 = ax2.bar(x + offsets[i], tokj_vals, WIDTH, label=mode,
                        color=MODE_PAL[mode], edgecolor="white", alpha=al)
        for bar, v in zip(bars1, toks_vals):
            if not np.isnan(v):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
        for bar, v in zip(bars2, tokj_vals):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
    ax1.set_title(f"Output Tok/s -- ctx={ctx_val}, gen={gen_val}", fontweight="bold")
    ax1.set_ylabel("Output tokens per second")
    ax2.set_title(f"Output Tok/J -- ctx={ctx_val}, gen={gen_val}", fontweight="bold")
    ax2.set_ylabel("Output tokens per joule")
    fig.suptitle(
        f"All Power Modes: ctx={ctx_val} tok prompt, gen={gen_val} tok output{title_suffix}",
        fontsize=13, fontweight="bold")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=11, loc="lower center",
               ncol=4, frameon=True, bbox_to_anchor=(0.5, 0),
               handlelength=2.0, handletextpad=0.8, columnspacing=1.2)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    save(fig, f"{fname_prefix}_ctx{ctx_val}_gen{gen_val}.png")


# ════════════════════════════════════════════════════════════════════════════════
print(f"\n── Main charts (canonical ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) ──")

# 1. Output Tok/s vs Prompt: 3×3 facet, canonical gen
faceted_line_chart(df, "tok_s", "Output Tok/s", CANONICAL_GEN, f"1_tok_s_vs_prompt_gen{CANONICAL_GEN}.png")

# 2. Output Tok/J vs Prompt: 3×3 facet, canonical gen
faceted_line_chart(df, "tok_j", "Output Tok/J", CANONICAL_GEN, f"2_tok_j_vs_prompt_gen{CANONICAL_GEN}.png")

# 3. Best Tok/J grouped bar
best = df.groupby(["model", "mode"])["tok_j"].max().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(22, 6.5))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    al = BACKEND_ALPHA.get(mode, 1.0)
    vals = [best.loc[m, mode] if (m in best.index and mode in best.columns
                                   and not pd.isna(best.loc[m, mode])) else 0
            for m in MODELS]
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", rotation=90, alpha=al)
ax.set_xticks(x)
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Best Output Tok/J per Model -- All Power Modes", fontweight="bold", fontsize=12)
ax.set_ylabel("Output Tok/J")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "3_best_tok_j_bar.png")

# 4. Average power grouped bar -- canonical ctx=2048, gen=256
canon = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
avg_pw = canon.groupby(["model", "mode"])["power_w"].median().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(22, 6.5))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    al = BACKEND_ALPHA.get(mode, 1.0)
    vals = [avg_pw.loc[m, mode] if (m in avg_pw.index and mode in avg_pw.columns
                                     and not pd.isna(avg_pw.loc[m, mode])) else 0
            for m in MODELS]
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}W", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
ax.set_xticks(x)
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Average Power Draw per Model -- All Power Modes (VDD_CPU_GPU_CV, ctx=2048 gen=256)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Power (W)")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "4_avg_power_bar.png")

# 5. TTFT p50 grouped bar -- canonical ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(22, 6.5))
x_bar = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    al = BACKEND_ALPHA.get(mode, 1.0)
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["ttft"].median() if not sub.empty else np.nan)
    bars = ax.bar(x_bar + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
ax.set_xticks(x_bar)
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"TTFT p50 by Power Mode (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("TTFT p50 (ms)")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "5_ttft_vs_prompt.png")

# 6. Speedup vs 15W baseline
toks_by_mode = df.groupby(["model", "mode"])["tok_s"].median().unstack("mode").reindex(MODELS)
fig, ax = plt.subplots(figsize=(22, 6.5))
cmp_modes = [m for m in ["7W", "25W", "MAXN", "7W-ollama", "15W-ollama", "25W-ollama", "MAXN-ollama"] if m in modes_avail]
for i, cmp_mode in enumerate(cmp_modes):
    speedups = []
    base_mode = "15W" if "15W" in toks_by_mode.columns else None
    for m in MODELS:
        base = toks_by_mode.loc[m, base_mode] if base_mode else None
        cmp  = toks_by_mode.loc[m, cmp_mode] if cmp_mode in toks_by_mode.columns else None
        if base and cmp and not pd.isna(base) and not pd.isna(cmp):
            speedups.append(cmp / base)
        else:
            speedups.append(np.nan)
    offs_i = np.linspace(-(len(cmp_modes) - 1) * WIDTH / 2, (len(cmp_modes) - 1) * WIDTH / 2, len(cmp_modes))
    bars = ax.bar(np.arange(len(MODELS)) + offs_i[i], speedups, WIDTH,
                  label=f"{cmp_mode} ÷ 15W", color=MODE_PAL[cmp_mode],
                  edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, speedups):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.axhline(1.0, color="grey", ls="--", lw=1.2, alpha=0.7, label="15W baseline")
ax.set_xticks(np.arange(len(MODELS)))
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title("Output Throughput Speedup vs 15W -- avg over all prompt × gen combos",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Tok/s ratio (mode ÷ 15W)")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "6_speedup_vs_15w.png")

# 8. ITL comparison: canonical ctx=2048, gen=256
fig, ax = plt.subplots(figsize=(22, 6.5))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    al = BACKEND_ALPHA.get(mode, 1.0)
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["itl"].median() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, alpha=al)
ax.set_xticks(x)
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Inter-Token Latency p50 -- All Power Modes (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("ITL p50 (ms)")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "8_itl_compare.png")

# 8b. Phase power heatmap: prefill + decode power -- models × modes (canonical cell)
for phase_col, phase_label, cmap in [
    ("prefill_power_w", "Prefill Power (W)", "OrRd"),
    ("decode_power_w",  "Decode Power (W)",  "Blues"),
]:
    matrix = []
    for m in MODELS:
        row_vals = []
        for mode in modes_avail:
            sub = df[(df.model == m) & (df["mode"] == mode) &
                     (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
            row_vals.append(sub[phase_col].median() if not sub.empty else np.nan)
        matrix.append(row_vals)
    phase_df = pd.DataFrame(matrix,
                            index=[MDL.get(m, m) for m in MODELS],
                            columns=modes_avail)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(phase_df, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": phase_label}, ax=ax)
    ax.set_title(f"{phase_label} -- All Models × All Power Modes\n(ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Power Mode")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fname = ("EP_prefill_power_heatmap_canonical.png" if "prefill" in phase_col
             else "EP_decode_power_heatmap_canonical.png")
    save(fig, fname)

# 9. Request latency (E2E) comparison
fig, ax = plt.subplots(figsize=(22, 6.5))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    al = BACKEND_ALPHA.get(mode, 1.0)
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) &
                 (df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["rl_p50"].median() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
ax.set_xticks(x)
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Request Latency (E2E) p50 -- All Power Modes (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN})",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Request latency p50 (ms)")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "10_request_latency_compare.png")

# 10. Prefill throughput
fig, ax = plt.subplots(figsize=(22, 6.5))
x = np.arange(len(MODELS))
for i, mode in enumerate(modes_avail):
    al = BACKEND_ALPHA.get(mode, 1.0)
    vals = []
    for m in MODELS:
        sub = df[(df.model == m) & (df["mode"] == mode) & (df.gen == CANONICAL_GEN)]
        vals.append(sub["prefill"].median() if not sub.empty else np.nan)
    bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                  color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
ax.set_xticks(x)
ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
ax.set_title(f"Prefill Throughput -- All Power Modes (gen={CANONICAL_GEN}, avg over all prompts)",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Prefill tok/s")
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "9_prefill_compare.png")

# 11. Canonical cell comparison: ctx=2048, gen=256
canonical_bar(CANONICAL_CTX, CANONICAL_GEN,
              fname_prefix="11_canonical_cell_comparison",
              title_suffix="  [canonical standard]")

# 12. Energy: total_J vs gen at all modes, ctx=2048
fig, ax = plt.subplots(figsize=(12, 6))
sub2048 = df[df.prompt == CANONICAL_CTX]
for model in MODELS:
    s = sub2048[sub2048.model == model & sub2048["mode"].isin(modes_avail)].sort_values("gen") if False else \
        sub2048[(sub2048.model == model)].sort_values("gen")
    if s.empty or s["total_j"].dropna().empty:
        continue
    ax.plot(s.gen, s.total_j, marker="s", ls="--", lw=2,
            color=MODEL_PAL[model], label=MDL.get(model, model), ms=8)
ax.set_title(f"Total Energy per Request vs Gen Length (ctx={CANONICAL_CTX})",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Gen length (tok)")
ax.set_ylabel("Total energy (J)")
ax.set_xticks(GEN_LENGTHS)
ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
save(fig, "E_total_energy_vs_gen_length.png")

# ── Speedup comparison chart: llama.cpp vs Ollama (canonical cell) ──────────
print("\n── Speedup comparison chart (llama.cpp vs Ollama) ──")
canon = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)].copy()
pairs = [("25W/15W","25W","15W"),("MAXN/15W","MAXN","15W"),("15W/7W","15W","7W"),
         ("25W/7W","25W","7W"),("MAXN/7W","MAXN","7W"),("MAXN/25W","MAXN","25W")]
n_models = len(MODELS)
W = 0.10

fig, ax = plt.subplots(figsize=(18, 7))
offs = np.linspace(-(n_models - 1) * W / 2, (n_models - 1) * W / 2, n_models)
for gi, (label, num_m, den_m) in enumerate(pairs):
    base_x = gi * 3
    for mi, model in enumerate(MODELS):
        num = canon[(canon.model==model)&(canon["mode"]==num_m)]["tok_s"].median()
        den = canon[(canon.model==model)&(canon["mode"]==den_m)]["tok_s"].median()
        v = num/den if (num and den and den>0) else 0
        ax.bar(base_x + offs[mi], v, W, color=MODEL_PAL[model], alpha=1.0,
               edgecolor="white", linewidth=0.5)
        num = canon[(canon.model==model)&(canon["mode"]==f"{num_m}-ollama")]["tok_s"].median()
        den = canon[(canon.model==model)&(canon["mode"]==f"{den_m}-ollama")]["tok_s"].median()
        v = num/den if (num and den and den>0) else 0
        ax.bar(base_x + 1 + offs[mi], v, W, color=MODEL_PAL[model], alpha=0.55,
               edgecolor="white", linewidth=0.5)
    ax.text(base_x + 0.5, -0.15, "llama.cpp", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.5, -0.15, "Ollama", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.0, -0.45, label, ha="center", va="top", fontsize=11, fontweight="bold")

ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
ax.set_xlim(-0.5, len(pairs) * 3 - 0.5)
ax.set_xticks([])
ax.set_ylabel("Speedup ratio (faster ÷ baseline)", fontsize=12)
ax.set_title("Output Throughput Speedup Ratios -- llama.cpp vs Ollama\ncanonical cell (ctx=2048, gen=256) · solid=llama.cpp · faded=Ollama",
             fontsize=13, pad=10)
handles = [mlines.Line2D([],[],color=MODEL_PAL[m],lw=3,label=MDL[m]) for m in MODELS]
spacer = mlines.Line2D([],[],color="none",label="")
style_h = [
    mlines.Line2D([],[],color="gray",lw=3,alpha=1.0,label="llama.cpp"),
    mlines.Line2D([],[],color="gray",lw=3,alpha=0.55,label="Ollama"),
]
ax.legend(handles=handles+[spacer]+style_h, loc="upper left", bbox_to_anchor=(1.01,1),
          fontsize=9, frameon=True, title="Model / Backend", title_fontsize=10)
fig.tight_layout()
save(fig, "12_speedup_ratios_comparison.png")

# ── TTFT speedup ratios bar chart (canonical cell) ─────────────────────────
print("\n── TTFT speedup ratios chart (llama.cpp vs Ollama) ──")
canon = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)].copy()
pairs = [("25W/15W","25W","15W"),("MAXN/15W","MAXN","15W"),("15W/7W","15W","7W"),
         ("25W/7W","25W","7W"),("MAXN/7W","MAXN","7W"),("MAXN/25W","MAXN","25W")]
n_models = len(MODELS)
W = 0.10

fig, ax = plt.subplots(figsize=(18, 7))
offs = np.linspace(-(n_models - 1) * W / 2, (n_models - 1) * W / 2, n_models)
for gi, (label, num_m, den_m) in enumerate(pairs):
    base_x = gi * 3
    for mi, model in enumerate(MODELS):
        num = canon[(canon.model==model)&(canon["mode"]==num_m)]["ttft"].median()
        den = canon[(canon.model==model)&(canon["mode"]==den_m)]["ttft"].median()
        v = den/num if (num and den and den>0) else 0
        ax.bar(base_x + offs[mi], v, W, color=MODEL_PAL[model], alpha=1.0,
               edgecolor="white", linewidth=0.5)
        num = canon[(canon.model==model)&(canon["mode"]==f"{num_m}-ollama")]["ttft"].median()
        den = canon[(canon.model==model)&(canon["mode"]==f"{den_m}-ollama")]["ttft"].median()
        v = den/num if (num and den and den>0) else 0
        ax.bar(base_x + 1 + offs[mi], v, W, color=MODEL_PAL[model], alpha=0.55,
               edgecolor="white", linewidth=0.5)
    ax.text(base_x + 0.5, -0.15, "llama.cpp", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.5, -0.15, "Ollama", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.0, -0.45, label, ha="center", va="top", fontsize=11, fontweight="bold")

ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
ax.set_xlim(-0.5, len(pairs) * 3 - 0.5)
ax.set_xticks([])
ax.set_ylabel("Speedup ratio (faster ÷ baseline)", fontsize=12)
ax.set_title("TTFT Speedup Ratios -- llama.cpp vs Ollama\ncanonical cell (ctx=2048, gen=256) · solid=llama.cpp · faded=Ollama · higher=faster prefill",
             fontsize=13, pad=10)
handles = [mlines.Line2D([],[],color=MODEL_PAL[m],lw=3,label=MDL[m]) for m in MODELS]
spacer = mlines.Line2D([],[],color="none",label="")
style_h = [
    mlines.Line2D([],[],color="gray",lw=3,alpha=1.0,label="llama.cpp"),
    mlines.Line2D([],[],color="gray",lw=3,alpha=0.55,label="Ollama"),
]
ax.legend(handles=handles+[spacer]+style_h, loc="upper left", bbox_to_anchor=(1.01,1),
          fontsize=9, frameon=True, title="Model / Backend", title_fontsize=10)
fig.tight_layout()
save(fig, "13_ttft_speedup_ratios.png")

# ── Decode time speedup ratios bar chart (canonical cell) ──────────────────
print("\n── Decode time speedup ratios chart (llama.cpp vs Ollama) ──")
canon = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)].copy()
# Decode time = RL_p50 - TTFT (both in ms)
canon = canon.copy()
canon["decode_time_ms"] = canon["rl_p50"] - canon["ttft"]

pairs = [("25W/15W","25W","15W"),("MAXN/15W","MAXN","15W"),("15W/7W","15W","7W"),
         ("25W/7W","25W","7W"),("MAXN/7W","MAXN","7W"),("MAXN/25W","MAXN","25W")]
n_models = len(MODELS)
W = 0.10

fig, ax = plt.subplots(figsize=(18, 7))
offs = np.linspace(-(n_models - 1) * W / 2, (n_models - 1) * W / 2, n_models)
for gi, (label, num_m, den_m) in enumerate(pairs):
    base_x = gi * 3
    for mi, model in enumerate(MODELS):
        num = canon[(canon.model==model)&(canon["mode"]==num_m)]["decode_time_ms"].median()
        den = canon[(canon.model==model)&(canon["mode"]==den_m)]["decode_time_ms"].median()
        v = den/num if (num and den and den>0) else 0
        ax.bar(base_x + offs[mi], v, W, color=MODEL_PAL[model], alpha=1.0,
               edgecolor="white", linewidth=0.5)
        num = canon[(canon.model==model)&(canon["mode"]==f"{num_m}-ollama")]["decode_time_ms"].median()
        den = canon[(canon.model==model)&(canon["mode"]==f"{den_m}-ollama")]["decode_time_ms"].median()
        v = den/num if (num and den and den>0) else 0
        ax.bar(base_x + 1 + offs[mi], v, W, color=MODEL_PAL[model], alpha=0.55,
               edgecolor="white", linewidth=0.5)
    ax.text(base_x + 0.5, -0.15, "llama.cpp", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.5, -0.15, "Ollama", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.0, -0.45, label, ha="center", va="top", fontsize=11, fontweight="bold")

ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
ax.set_xlim(-0.5, len(pairs) * 3 - 0.5)
ax.set_xticks([])
ax.set_ylabel("Speedup ratio (faster ÷ baseline)", fontsize=12)
ax.set_title("Decode Time Speedup Ratios -- llama.cpp vs Ollama\ncanonical cell (ctx=2048, gen=256) · solid=llama.cpp · faded=Ollama · higher=faster decode",
             fontsize=13, pad=10)
handles = [mlines.Line2D([],[],color=MODEL_PAL[m],lw=3,label=MDL[m]) for m in MODELS]
spacer = mlines.Line2D([],[],color="none",label="")
style_h = [
    mlines.Line2D([],[],color="gray",lw=3,alpha=1.0,label="llama.cpp"),
    mlines.Line2D([],[],color="gray",lw=3,alpha=0.55,label="Ollama"),
]
ax.legend(handles=handles+[spacer]+style_h, loc="upper left", bbox_to_anchor=(1.01,1),
          fontsize=9, frameon=True, title="Model / Backend", title_fontsize=10)
fig.tight_layout()
save(fig, "14_decode_time_speedup_ratios.png")

# ── Prefill throughput speedup ratios bar chart (canonical cell) ────────────
print("\n── Prefill throughput speedup ratios chart (llama.cpp vs Ollama) ──")
canon = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)].copy()
pairs = [("25W/15W","25W","15W"),("MAXN/15W","MAXN","15W"),("15W/7W","15W","7W"),
         ("25W/7W","25W","7W"),("MAXN/7W","MAXN","7W"),("MAXN/25W","MAXN","25W")]
n_models = len(MODELS)
W = 0.10

fig, ax = plt.subplots(figsize=(18, 7))
offs = np.linspace(-(n_models - 1) * W / 2, (n_models - 1) * W / 2, n_models)
for gi, (label, num_m, den_m) in enumerate(pairs):
    base_x = gi * 3
    for mi, model in enumerate(MODELS):
        num = canon[(canon.model==model)&(canon["mode"]==num_m)]["prefill"].median()
        den = canon[(canon.model==model)&(canon["mode"]==den_m)]["prefill"].median()
        v = num/den if (num and den and den>0) else 0
        ax.bar(base_x + offs[mi], v, W, color=MODEL_PAL[model], alpha=1.0,
               edgecolor="white", linewidth=0.5)
        num = canon[(canon.model==model)&(canon["mode"]==f"{num_m}-ollama")]["prefill"].median()
        den = canon[(canon.model==model)&(canon["mode"]==f"{den_m}-ollama")]["prefill"].median()
        v = num/den if (num and den and den>0) else 0
        ax.bar(base_x + 1 + offs[mi], v, W, color=MODEL_PAL[model], alpha=0.55,
               edgecolor="white", linewidth=0.5)
    ax.text(base_x + 0.5, -0.15, "llama.cpp", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.5, -0.15, "Ollama", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(base_x + 1.0, -0.45, label, ha="center", va="top", fontsize=11, fontweight="bold")

ax.axhline(1.0, color="gray", lw=1, ls=":", alpha=0.7)
ax.set_xlim(-0.5, len(pairs) * 3 - 0.5)
ax.set_xticks([])
ax.set_ylabel("Speedup ratio (faster ÷ baseline)", fontsize=12)
ax.set_title("Prefill Throughput Speedup Ratios -- llama.cpp vs Ollama\ncanonical cell (ctx=2048, gen=256) · solid=llama.cpp · faded=Ollama · higher=faster prefill",
             fontsize=13, pad=10)
handles = [mlines.Line2D([],[],color=MODEL_PAL[m],lw=3,label=MDL[m]) for m in MODELS]
spacer = mlines.Line2D([],[],color="none",label="")
style_h = [
    mlines.Line2D([],[],color="gray",lw=3,alpha=1.0,label="llama.cpp"),
    mlines.Line2D([],[],color="gray",lw=3,alpha=0.55,label="Ollama"),
]
ax.legend(handles=handles+[spacer]+style_h, loc="upper left", bbox_to_anchor=(1.01,1),
          fontsize=9, frameon=True, title="Model / Backend", title_fontsize=10)
fig.tight_layout()
save(fig, "18_prefill_tput_speedup_ratios.png")

# ── Appendix E: llamacpp/ollama ratio tables (written to ratio_tables.md) ──────
print("\n── Ratio tables (llamacpp ÷ ollama) ──")
RATIO_PAIRS = [("7W", "7W-ollama"), ("15W", "15W-ollama"), ("25W", "25W-ollama"), ("MAXN", "MAXN-ollama")]
RATIO_METRICS = [
    ("tok_s",  "Tok/s",  ".2f"),
    ("tok_j",  "Tok/J",  ".2f"),
    ("ttft",   "TTFT",   ".2f"),
    ("itl",    "ITL",    ".2f"),
    ("power_w","Power",  ".2f"),
]
can = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
ratio_md_lines = [
    "<!-- ratio_tables: llamacpp ÷ ollama at canonical cell (ctx=2048, gen=256) -->",
    "<!-- values >1 mean llamacpp is faster/more efficient/higher power -->",
    "",
]
for lc_mode, ol_mode in RATIO_PAIRS:
    if lc_mode not in modes_avail or ol_mode not in modes_avail:
        continue
    cols = ["Model"] + [f"{m} ratio" for m, _, _ in RATIO_METRICS]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for model in MODELS:
        lc_row = can[(can.model == model) & (can["mode"] == lc_mode)]
        ol_row = can[(can.model == model) & (can["mode"] == ol_mode)]
        cells = [MDL.get(model, model)]
        for met, _, fmt in RATIO_METRICS:
            lc_v = lc_row[met].median() if not lc_row.empty else np.nan
            ol_v = ol_row[met].median() if not ol_row.empty else np.nan
            if pd.isna(lc_v) or pd.isna(ol_v) or ol_v == 0:
                cells.append("--")
            else:
                ratio = lc_v / ol_v
                cells.append(format(ratio, fmt) + "×")
        rows.append("| " + " | ".join(cells) + " |")
    ratio_md_lines += [
        f"### {lc_mode} vs {ol_mode}: llamacpp ÷ ollama",
        "",
    ] + rows + [""]
    print(f"  ratio table: {lc_mode} / {ol_mode}")

ratio_path = OUT_DIR / "ratio_tables.md"
ratio_path.write_text("\n".join(ratio_md_lines))
print(f"  written {ratio_path}")

# ── line_chart_by_model_vs_mode: x=power mode, lines=models ──────────────────
# Used for Appendix D (Prefill/Decode/Total tok/J). One chart per gen length.
# ctx fixed at CANONICAL_CTX (2048). Solid=llama.cpp, dashed=Ollama.

LC_MODES = ["7W", "15W", "25W", "MAXN"]
OL_MODES = ["7W-ollama", "15W-ollama", "25W-ollama", "MAXN-ollama"]
MODE_X   = np.arange(len(LC_MODES))
MODE_XLAB = LC_MODES

def line_chart_by_model_vs_mode(df_in, y_field, y_label, gen_val, ctx_val, fname):
    sub = df_in[(df_in.gen == gen_val) & (df_in.prompt == ctx_val) &
                (df_in["mode"].isin(modes_avail))]
    if sub.empty or sub[y_field].dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(13, 7))

    for model in MODELS:
        color = MODEL_PAL[model]
        label = MDL[model]

        lc_vals = [sub[(sub.model == model) & (sub["mode"] == m)][y_field].median()
                   if not sub[(sub.model == model) & (sub["mode"] == m)].empty else None
                   for m in LC_MODES]
        ol_vals = [sub[(sub.model == model) & (sub["mode"] == m)][y_field].median()
                   if not sub[(sub.model == model) & (sub["mode"] == m)].empty else None
                   for m in OL_MODES]

        lc_pairs = [(x, y) for x, y in zip(MODE_X, lc_vals) if y is not None and not pd.isna(y)]
        ol_pairs = [(x, y) for x, y in zip(MODE_X, ol_vals) if y is not None and not pd.isna(y)]

        if lc_pairs:
            xx, yy = zip(*lc_pairs)
            ax.plot(list(xx), list(yy), color=color, ls="-", lw=2.5, marker="o", ms=8, label=label)
        if ol_pairs:
            xx, yy = zip(*ol_pairs)
            ax.plot(list(xx), list(yy), color=color, ls="--", lw=2.0, marker="s", ms=7, alpha=0.65)

    ax.set_xticks(MODE_X)
    ax.set_xticklabels(MODE_XLAB, fontsize=13)
    ax.set_xlabel("Power Mode", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(
        f"{y_label} across power modes -- all models\n"
        f"gen={gen_val} tok · ctx={ctx_val} tok · solid=llama.cpp · dashed=Ollama",
        fontsize=13, pad=10,
    )
    ax.tick_params(axis="y", labelsize=11)

    spacer = mlines.Line2D([], [], color="none", label="")
    style_handles = [
        mlines.Line2D([], [], color="gray", ls="-",  lw=2.5, marker="o", ms=6, label="llama.cpp"),
        mlines.Line2D([], [], color="gray", ls="--", lw=2.0, marker="s", ms=6, alpha=0.65, label="Ollama"),
    ]
    ax.legend(
        handles=model_legend_handles(MODELS) + [spacer] + style_handles,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
        fontsize=10, frameon=True, ncol=1,
        title="Model  /  Backend", title_fontsize=10,
        handlelength=2.2, handletextpad=0.7,
    )
    fig.tight_layout()
    save(fig, fname)


# ── Main body Figures 1 and 3: tok/s and tok/J vs power mode ──────────────────
line_chart_by_model_vs_mode(df, "tok_s", "Output Tok/s", CANONICAL_GEN, CANONICAL_CTX,
                            f"1_tok_s_vs_mode_gen{CANONICAL_GEN}.png")
line_chart_by_model_vs_mode(df, "tok_j", "Output Tok/J", CANONICAL_GEN, CANONICAL_CTX,
                            f"3_tok_j_vs_mode_gen{CANONICAL_GEN}.png")

# ── Comparison line charts (EF, EG, EH, EI) ──────────────────────────────────
print("\n── Comparison line charts ──")

# Appendix D.1 -- Prefill tok/J: x=mode, lines=models, one chart per gen length
# Appendix D.2 -- Decode tok/J:  same layout
# Appendix D.3 -- Total tok/J:   same layout
for gen_val in GEN_LENGTHS:
    line_chart_by_model_vs_mode(df, "prefill_tokj", "Prefill tok/J", gen_val, CANONICAL_CTX,
                                f"EF_prefill_tokj_vs_mode_gen{gen_val}.png")
    line_chart_by_model_vs_mode(df, "decode_tokj",  "Decode tok/J",  gen_val, CANONICAL_CTX,
                                f"EF_decode_tokj_vs_mode_gen{gen_val}.png")
    line_chart_by_model_vs_mode(df, "total_tokj",   "Total tok/J",   gen_val, CANONICAL_CTX,
                                f"EF_total_tokj_vs_mode_gen{gen_val}.png")

# Appendix E -- Request latency: x=power mode, lines=model colors
for gen_val in GEN_LENGTHS:
    line_chart_by_model_vs_mode(df, "rl_p50", "Request latency (ms)", gen_val, CANONICAL_CTX,
                                f"EF_req_latency_vs_mode_gen{gen_val}.png")

# Appendix F -- TTFT: x=power mode, lines=model colors
for gen_val in [64, 256]:  # TTFT independent of gen, only 64 and 256 needed
    line_chart_by_model_vs_mode(df, "ttft", "TTFT p50 (ms)", gen_val, CANONICAL_CTX,
                                f"EG_ttft_vs_mode_gen{gen_val}.png")

# Appendix G -- ITL: x=power mode, lines=model colors
for gen_val in GEN_LENGTHS:
    line_chart_by_model_vs_mode(df, "itl", "ITL p50 (ms)", gen_val, CANONICAL_CTX,
                                f"EH_itl_vs_mode_gen{gen_val}.png")

# Keep the faceted prompt-length charts for references in main body (section 2.3, 3.5)
for gen_val in GEN_LENGTHS:
    prefix = "EF_req_latency" if gen_val < CANONICAL_GEN else "22a_request_latency"
    faceted_line_chart(df, "rl_p50", "Request latency (ms)", gen_val,
                       f"{prefix}_vs_prompt_gen{gen_val}.png")
for gen_val in [64, 256]:
    faceted_line_chart(df, "ttft", "TTFT p50 (ms)", gen_val,
                       f"EG_ttft_vs_prompt_gen{gen_val}.png")
for gen_val in GEN_LENGTHS:
    faceted_line_chart(df, "itl", "ITL p50 (ms)", gen_val,
                       f"EH_itl_vs_prompt_gen{gen_val}.png")
    faceted_line_chart(df, "prefill", "Prefill tok/s", gen_val,
                       f"EI_prefill_tput_vs_prompt_gen{gen_val}.png")

for ctx_val in PROMPT_LENGTHS:
    facet_vs_gen(ctx_val, "ttft",    "TTFT p50 (ms)", f"EG_ttft_vs_gen_ctx{ctx_val}.png")
    facet_vs_gen(ctx_val, "itl",     "ITL p50 (ms)",  f"EH_itl_vs_gen_ctx{ctx_val}.png")
    facet_vs_gen(ctx_val, "prefill", "Prefill tok/s", f"EI_prefill_tput_vs_gen_ctx{ctx_val}.png")

# ── Alias charts: mode-x versions for main body Figures 7a/b/c and 10a ───────
import shutil
for src, dst in [
    ("EF_prefill_tokj_vs_mode_gen256.png",  "22e_prefill_tokj_vs_mode_gen256.png"),
    ("EF_decode_tokj_vs_mode_gen256.png",   "22f_decode_tokj_vs_mode_gen256.png"),
    ("EF_total_tokj_vs_mode_gen256.png",    "22g_total_tokj_vs_mode_gen256.png"),
    ("EH_itl_vs_mode_gen256.png",          "EH_itl_vs_prompt_gen256.png"),
]:
    src_p = OUT_DIR / src
    dst_p = OUT_DIR / dst
    if src_p.exists():
        shutil.copy2(src_p, dst_p)
        print(f"  aliased {src} → {dst}")

# ── Old alias section ────────────────────────────────────────────────────────

# ── E_mj_per_output_token: mJ per output token at canonical cell ──────────────
mj_data = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)].copy()
mj_data = mj_data[mj_data["tok_j"].notna()]
if not mj_data.empty:
    mj_data["mj_per_tok"] = 1000.0 / mj_data["tok_j"]
    fig, ax = plt.subplots(figsize=(22, 6.5))
    x = np.arange(len(MODELS))
    for i, mode in enumerate(modes_avail):
        vals = []
        for m in MODELS:
            sub = mj_data[(mj_data.model == m) & (mj_data["mode"] == mode)]
            vals.append(sub["mj_per_tok"].median() if not sub.empty else np.nan)
        al = BACKEND_ALPHA.get(mode, 1.0)
        bars = ax.bar(x + offsets[i], vals, WIDTH, label=mode,
                      color=MODE_PAL[mode], edgecolor="white", linewidth=0.8, alpha=al)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90, alpha=al)
    ax.set_xticks(x)
    ax.set_xticklabels([MDL.get(m, m) for m in MODELS], rotation=20, ha="right", fontsize=9)
    ax.set_title(f"Decode Energy per Output Token (mJ) -- ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("mJ per output token (lower = better)")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    save(fig, "E_mj_per_output_token.png")

print(f"\nAll charts saved to {OUT_DIR}")
