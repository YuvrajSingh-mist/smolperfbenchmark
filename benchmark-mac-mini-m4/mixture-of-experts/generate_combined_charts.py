#!/usr/bin/env python3
"""Generate comparison charts for the Mac Mini M4 MoE benchmark.

Models: granite4-h-tiny, lfm2.5-8b-a1b, smallthinker-4b-a0.6b, trinity-nano
Sweep: prompt in {256, 512, 1024, 2048, 4096} tok x gen in {256, 512, 1024} tok x 20 reqs/combo
Canonical cell: ctx=30720, gen=1024
Power source: macOS `powermetrics` (Combined CPU+GPU+ANE rail), NOT tegrastats — there is no
power-mode axis on Apple Silicon (no nvpmodel), so every chart here compares models directly
rather than faceting by power mode.

Power/tok-J math is reused verbatim from the benchmark harness's own report generator
(/tmp/blog_report_mac.py, embedded in benchmark-moe.sh) so the numbers here match report.md
and BLOG.md exactly: phase power comes from powermetrics samples classified into exact
prefill/decode windows using per-request nanosecond timestamps from profile_export.jsonl,
and output_tok_J = OSL_p50 / (decode_power_W * p50_decode_s) — decode energy only.

Charts saved to artifacts/charts/ (referenced by BLOG.md).
"""

import json, re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ROOT         = Path(__file__).parent
LLAMACPP_DIR = ROOT / "artifacts/mac-m4-moe-20260704-0115/llamacpp"
MLXLM_DIR    = ROOT / "artifacts/mac-m4-moe-20260712-1733/mlxlm"
RUN_DIR      = LLAMACPP_DIR  # kept for readability below; llama.cpp is the "primary" 6-model run

OUT_DIR = ROOT / "artifacts/charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "granite4-h-tiny",
    "lfm2.5-8b-a1b",
    "smallthinker-4b-a0.6b",
    "trinity-nano",
    "gemma4-e2b",
    "gemma4-e4b",
]
# Only these 3 have published 4-bit MLX conversions that load under mlx_lm.server
# (smallthinker has no MLX weights; the two gemma4 VLMs need mlx_vlm.server instead
# — see the "Backend coverage caveat" in BLOG.md).
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

# Line/point convention throughout (matches the Jetson benchmark post): solid
# lines / filled markers = llama.cpp, dashed lines / open (hollow) markers =
# MLX-LM. Bars follow the same idea: solid fill = llama.cpp, hatched = MLX-LM.
BACKENDS = ["llamacpp", "mlxlm"]
BACKEND_LABEL = {"llamacpp": "llama.cpp", "mlxlm": "MLX-LM"}
BACKEND_LS    = {"llamacpp": "-", "mlxlm": "--"}
BACKEND_HATCH = {"llamacpp": None, "mlxlm": "//"}
BACKEND_ALPHA = {"llamacpp": 1.0, "mlxlm": 0.85}

# Prompt lengths span 256 -> 30720 (>100x) and are NOT evenly spaced, so plotting
# them on a linear numeric x-axis crams 256/512/1024/2048/4096 into a sliver on
# the left and collides their tick labels. Every line chart plots against this
# fixed categorical x position instead, with compact "K" labels.
PROMPT_XPOS = {p: i for i, p in enumerate(PROMPT_LENGTHS)}
PROMPT_LABELS = [f"{p // 1024}K" if p >= 1024 else str(p) for p in PROMPT_LENGTHS]
PROMPT_LABEL_MAP = dict(zip(PROMPT_LENGTHS, PROMPT_LABELS))

CANONICAL_CTX = 30720
CANONICAL_GEN = 1024

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


# ── powermetrics loader (ported from /tmp/blog_report_mac.py's load_powermetrics) ──
# Combined (CPU+GPU+ANE) rail, one record per "Sampled system activity" block.
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


# ── phase-separated tok/J (ported from /tmp/blog_report_mac.py's compute_tok_j) ──
# tok/J = OSL_p50 / (decode_power_W x p50_decode_s) — decode energy only.
# prefill/decode windows come from profile_export.jsonl per-request ns timestamps;
# falls back to TTFT_p50 + RL_p50 timeline reconstruction if jsonl is missing.
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
                isl     = pct("input_sequence_length", "p50")
                osl     = pct("output_sequence_length", "p50")

                pm_records = load_powermetrics(combo_dir / "powermetrics.log")
                ph = compute_phase_power(combo_dir, pm_records, d)

                rows.append(dict(
                    backend=backend, model=model, prompt=ctx, gen=gen,
                    tok_s=tok_s, ttft=ttft, itl=itl, prefill=prefill, rl_p50=rl_p50,
                    isl=isl, osl=osl,
                    total_pw=ph["total_pw"], prefill_pw=ph["prefill_pw"], decode_pw=ph["decode_pw"],
                    tok_j=ph["tok_j"], prefill_tok_j=ph["prefill_tok_j"], total_tok_j=ph["total_tok_j"],
                ))
    return rows


rows = load_run(LLAMACPP_DIR, MODELS, "llamacpp") + load_run(MLXLM_DIR, MLXLM_MODELS, "mlxlm")

df = pd.DataFrame(rows)
df_lc  = df[df.backend == "llamacpp"]
df_mlx = df[df.backend == "mlxlm"]
print(f"Loaded {len(df)} rows across {df['model'].nunique()} models, {df['backend'].nunique()} backends")
print(f"llama.cpp: {len(df_lc)} rows, {sorted(df_lc['model'].unique())}")
print(f"MLX-LM:    {len(df_mlx)} rows, {sorted(df_mlx['model'].unique())}")

x = np.arange(len(MODELS))


# ═════════════════════════════ Line charts (per gen length) ═════════════════════════════
# Both backends on every chart: solid filled markers = llama.cpp (all 6 models),
# dashed hollow markers = MLX-LM (3 shared models, same color as llama.cpp's line
# for that model) — same convention as the Jetson llama.cpp-vs-Ollama post.
def line_chart_all_models(y_field, y_label, gen_val, fname, y_scale=None):
    data = df[df.gen == gen_val][y_field].dropna()
    if data.empty:
        return
    y_max = y_scale if y_scale else data.max() * 1.15
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for model in MODELS:
        s = df_lc[(df_lc.model == model) & (df_lc.gen == gen_val)].sort_values("prompt")
        if s.empty:
            continue
        xpos = s.prompt.map(PROMPT_XPOS)
        ax.plot(xpos, s[y_field], marker=MODEL_MARKER[model], lw=2, ls=BACKEND_LS["llamacpp"],
                color=MODEL_PAL[model], label=f"{MDL_FLAT[model]} (llama.cpp)", ms=8,
                markerfacecolor=MODEL_PAL[model], zorder=3)
    for model in MLXLM_MODELS:
        s = df_mlx[(df_mlx.model == model) & (df_mlx.gen == gen_val)].sort_values("prompt")
        if s.empty:
            continue
        xpos = s.prompt.map(PROMPT_XPOS)
        ax.plot(xpos, s[y_field], marker=MODEL_MARKER[model], lw=2, ls=BACKEND_LS["mlxlm"],
                color=MODEL_PAL[model], label=f"{MDL_FLAT[model]} (MLX-LM)", ms=8,
                markerfacecolor="white", markeredgecolor=MODEL_PAL[model], markeredgewidth=1.6,
                alpha=BACKEND_ALPHA["mlxlm"], zorder=2)
    ax.set_title(f"{y_label} vs Prompt Length (gen={gen_val} tok) — llama.cpp (solid) vs MLX-LM (dashed)",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Prompt (tok)")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(PROMPT_LENGTHS)))
    ax.set_xticklabels(PROMPT_LABELS)
    ax.set_xlim(-0.3, len(PROMPT_LENGTHS) - 0.7)
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, ncol=1)
    plt.tight_layout()
    save(fig, fname)


print("\n-- Line charts (both backends) --")
for gen_val in GEN_LENGTHS:
    line_chart_all_models("tok_s", "Output Tok/s", gen_val, f"1_tok_s_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("tok_j", "Output Tok/J", gen_val, f"2_tok_j_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("ttft", "TTFT p50 (ms)", gen_val, f"EG_ttft_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("itl", "ITL p50 (ms)", gen_val, f"EH_itl_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("prefill", "Prefill Tok/s", gen_val, f"EI_prefill_tput_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("rl_p50", "Request Latency p50 (ms)", gen_val, f"EA_request_latency_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("total_tok_j", "Total Tok/J", gen_val, f"22g_total_tokj_vs_prompt_gen{gen_val}.png")
    line_chart_all_models("prefill_tok_j", "Prefill Tok/J", gen_val, f"22e_prefill_tokj_vs_prompt_gen{gen_val}.png")


# ═════════════════════════════ Bar charts (single value per model, both backends) ═════════════════════════════
# Solid fill = llama.cpp (all 6 models); hatched fill = MLX-LM (3 shared models
# only — the other 3 models simply have no second bar, matching the "Backend
# coverage caveat" in BLOG.md).
def bar_chart_dual(values_lc, values_mlx, ylabel, title, fname, fmt="{:.2f}",
                    combo_lc=None, combo_mlx=None):
    # combo_lc / combo_mlx (optional): {model: "ctx/gen"} label of which cell in the
    # sweep produced that bar's value - drawn as a smaller line under the value, so
    # "best across all combos" bars don't leave the reader guessing which combo won.
    combo_lc = combo_lc or {}
    combo_mlx = combo_mlx or {}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    w = 0.36
    vals_lc  = [values_lc.get(m, np.nan) for m in MODELS]
    vals_mlx = [values_mlx.get(m, np.nan) for m in MODELS]
    cols = [MODEL_PAL[m] for m in MODELS]
    has_mlx = any(not (v is None or (isinstance(v, float) and np.isnan(v))) for v in vals_mlx)
    off = w / 2 if has_mlx else 0
    bars_lc = ax.bar(x - off, vals_lc, w, color=cols, edgecolor="white", linewidth=0.8,
                      label="llama.cpp")
    for bar, v, m in zip(bars_lc, vals_lc, MODELS):
        if not (v is None or (isinstance(v, float) and np.isnan(v))):
            cx = bar.get_x() + bar.get_width() / 2
            ax.annotate(fmt.format(v), (cx, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            if m in combo_lc:
                ax.annotate(f"@ {combo_lc[m]}", (cx, v), xytext=(0, 16), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color="dimgray")
    if has_mlx:
        bars_mlx = ax.bar(x + off, vals_mlx, w, color=cols, edgecolor="black", linewidth=0.9,
                           hatch="//", alpha=BACKEND_ALPHA["mlxlm"], label="MLX-LM")
        for bar, v, m in zip(bars_mlx, vals_mlx, MODELS):
            if not (v is None or (isinstance(v, float) and np.isnan(v))):
                cx = bar.get_x() + bar.get_width() / 2
                ax.annotate(fmt.format(v), (cx, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
                if m in combo_mlx:
                    ax.annotate(f"@ {combo_mlx[m]}", (cx, v), xytext=(0, 16), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel)
    if combo_lc or combo_mlx:
        all_vals = [v for v in vals_lc + vals_mlx if not (v is None or (isinstance(v, float) and np.isnan(v)))]
        if all_vals:
            ax.set_ylim(0, max(all_vals) * 1.22)
    if has_mlx:
        ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, fname)


print("\n-- Bar charts (both backends) --")

# 3. Best output tok/J per model (search across all combos)
best_tokj_lc  = df_lc.groupby("model")["tok_j"].max().to_dict()
best_tokj_mlx = df_mlx.groupby("model")["tok_j"].max().to_dict()


def best_combo_labels(df_sub):
    # Which (prompt, gen) cell produced each model's max tok_j - "ctx/gen" label,
    # e.g. "256/512", so a "best across all combos" bar says which combo won.
    sub = df_sub.dropna(subset=["tok_j"])
    if sub.empty:
        return {}
    idx = sub.groupby("model")["tok_j"].idxmax()
    out = {}
    for model, i in idx.items():
        row = sub.loc[i]
        ctx_label = PROMPT_LABEL_MAP.get(int(row.prompt), str(int(row.prompt)))
        out[model] = f"{ctx_label}/{int(row.gen)}"
    return out


combo_lc  = best_combo_labels(df_lc)
combo_mlx = best_combo_labels(df_mlx)
bar_chart_dual(best_tokj_lc, best_tokj_mlx, "Output Tok/J",
               "Best Output Tok/J per Model (searched across all combos) — llama.cpp vs MLX-LM",
               "3_best_tok_j_bar.png", combo_lc=combo_lc, combo_mlx=combo_mlx)

# 4. Average total power per model
avg_pw_lc  = df_lc.groupby("model")["total_pw"].mean().to_dict()
avg_pw_mlx = df_mlx.groupby("model")["total_pw"].mean().to_dict()
bar_chart_dual(avg_pw_lc, avg_pw_mlx, "Power (W)",
               "Average Total Power per Model (Combined CPU+GPU+ANE) — llama.cpp vs MLX-LM",
               "4_avg_power_bar.png", fmt="{:.2f}W")

# canonical-cell subsets, per backend
canon    = df[(df.prompt == CANONICAL_CTX) & (df.gen == CANONICAL_GEN)]
canon_lc  = canon[canon.backend == "llamacpp"]
canon_mlx = canon[canon.backend == "mlxlm"]

# 5. TTFT at canonical cell
ttft_canon_lc  = {r.model: r.ttft for r in canon_lc.itertuples()}
ttft_canon_mlx = {r.model: r.ttft for r in canon_mlx.itertuples()}
bar_chart_dual(ttft_canon_lc, ttft_canon_mlx, "TTFT p50 (ms)",
               f"TTFT p50 by Model (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
               "5_ttft_vs_prompt.png", fmt="{:.0f}")

# 8. ITL at canonical cell
itl_canon_lc  = {r.model: r.itl for r in canon_lc.itertuples()}
itl_canon_mlx = {r.model: r.itl for r in canon_mlx.itertuples()}
bar_chart_dual(itl_canon_lc, itl_canon_mlx, "ITL p50 (ms)",
               f"Inter-Token Latency p50 by Model (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
               "8_itl_compare.png", fmt="{:.1f}")

# 9. Prefill throughput (canonical gen, avg over all prompts)
prefill_avg_lc  = df_lc[df_lc.gen == CANONICAL_GEN].groupby("model")["prefill"].mean().to_dict()
prefill_avg_mlx = df_mlx[df_mlx.gen == CANONICAL_GEN].groupby("model")["prefill"].mean().to_dict()
bar_chart_dual(prefill_avg_lc, prefill_avg_mlx, "Prefill Tok/s",
               f"Prefill Throughput by Model (gen={CANONICAL_GEN}, avg over all prompts) — llama.cpp vs MLX-LM",
               "9_prefill_compare.png", fmt="{:.0f}")

# 10. Request latency (E2E) at canonical cell
rl_canon_lc  = {r.model: r.rl_p50 for r in canon_lc.itertuples()}
rl_canon_mlx = {r.model: r.rl_p50 for r in canon_mlx.itertuples()}
bar_chart_dual(rl_canon_lc, rl_canon_mlx, "Request Latency p50 (ms)",
               f"Request Latency (E2E) p50 by Model (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
               "10_request_latency_compare.png", fmt="{:.0f}")

# EP. Prefill vs decode power at canonical cell (grouped bar, both backends side by side per model)
fig, ax = plt.subplots(figsize=(15, 6.5))
n = len(MODELS)
w = 0.19
prefill_lc  = [canon_lc[canon_lc.model == m]["prefill_pw"].median() for m in MODELS]
decode_lc   = [canon_lc[canon_lc.model == m]["decode_pw"].median() for m in MODELS]
prefill_mlx = [canon_mlx[canon_mlx.model == m]["prefill_pw"].median() if m in MLXLM_MODELS else np.nan for m in MODELS]
decode_mlx  = [canon_mlx[canon_mlx.model == m]["decode_pw"].median() if m in MLXLM_MODELS else np.nan for m in MODELS]
bars1 = ax.bar(x - 1.5 * w, prefill_lc, w, label="Prefill W (llama.cpp)", color="#FF9800", edgecolor="white")
bars2 = ax.bar(x - 0.5 * w, decode_lc,  w, label="Decode W (llama.cpp)",  color="#2196F3", edgecolor="white")
bars3 = ax.bar(x + 0.5 * w, prefill_mlx, w, label="Prefill W (MLX-LM)", color="#FF9800", edgecolor="black", hatch="//", alpha=0.85)
bars4 = ax.bar(x + 1.5 * w, decode_mlx,  w, label="Decode W (MLX-LM)",  color="#2196F3", edgecolor="black", hatch="//", alpha=0.85)
for bars in (bars1, bars2, bars3, bars4):
    for bar in bars:
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_title(f"Prefill vs Decode Power by Model (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
             fontweight="bold", fontsize=12)
ax.set_ylabel("Power (W)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
save(fig, "EP_prefill_decode_power_canonical.png")

# 11. Canonical cell comparison: tok/s and tok/J side by side, both backends
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
toks_lc  = [canon_lc[canon_lc.model == m]["tok_s"].median() for m in MODELS]
tokj_lc  = [canon_lc[canon_lc.model == m]["tok_j"].median() for m in MODELS]
toks_mlx = [canon_mlx[canon_mlx.model == m]["tok_s"].median() if m in MLXLM_MODELS else np.nan for m in MODELS]
tokj_mlx = [canon_mlx[canon_mlx.model == m]["tok_j"].median() if m in MLXLM_MODELS else np.nan for m in MODELS]
cols = [MODEL_PAL[m] for m in MODELS]
w2 = 0.34
bars1a = ax1.bar(x - w2 / 2, toks_lc, w2, color=cols, edgecolor="white", label="llama.cpp")
bars1b = ax1.bar(x + w2 / 2, toks_mlx, w2, color=cols, edgecolor="black", hatch="//", alpha=0.85, label="MLX-LM")
bars2a = ax2.bar(x - w2 / 2, tokj_lc, w2, color=cols, edgecolor="white", label="llama.cpp")
bars2b = ax2.bar(x + w2 / 2, tokj_mlx, w2, color=cols, edgecolor="black", hatch="//", alpha=0.85, label="MLX-LM")
for bar, v in zip(bars1a, toks_lc):
    if not (v is None or np.isnan(v)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, v in zip(bars1b, toks_mlx):
    if not (v is None or np.isnan(v)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, v in zip(bars2a, tokj_lc):
    if not (v is None or np.isnan(v)):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, v in zip(bars2b, tokj_mlx):
    if not (v is None or np.isnan(v)):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for ax in (ax1, ax2):
    ax.set_xticks(x)
    ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=8)
ax1.set_title(f"Output Tok/s — ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}", fontweight="bold")
ax1.set_ylabel("Output tokens per second")
ax2.set_title(f"Output Tok/J — ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}", fontweight="bold")
ax2.set_ylabel("Output tokens per joule")
fig.suptitle(f"MoE Models on Mac Mini M4 — Canonical Cell: ctx={CANONICAL_CTX} tok prompt, gen={CANONICAL_GEN} tok output — llama.cpp vs MLX-LM",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, f"11_canonical_cell_comparison_ctx{CANONICAL_CTX}_gen{CANONICAL_GEN}.png")

# E_mj_per_output_token: mJ/output_tok at canonical cell
def mj_from_canon(canon_sub, models):
    out = {}
    for m in models:
        sub = canon_sub[canon_sub.model == m]
        if not sub.empty and sub["tok_j"].notna().any():
            tj = sub["tok_j"].median()
            out[m] = 1000.0 / tj if tj and tj > 0 else np.nan
    return out

mj_vals_lc  = mj_from_canon(canon_lc, MODELS)
mj_vals_mlx = mj_from_canon(canon_mlx, MLXLM_MODELS)
bar_chart_dual(mj_vals_lc, mj_vals_mlx, "mJ per output token",
               f"Decode Energy per Output Token (ctx={CANONICAL_CTX}, gen={CANONICAL_GEN}) — llama.cpp vs MLX-LM",
               "E_mj_per_output_token.png", fmt="{:.0f}mJ")

# Prefill/decode energy split (grouped stacked bar, % of total request energy, both backends)
def energy_split_pct(df_sub, models):
    prefill_pct, decode_pct = [], []
    for m in models:
        sub = df_sub[df_sub.model == m]
        pfj = (sub["prefill_pw"] * sub["ttft"] / 1000.0)
        dcj = (sub["decode_pw"] * (sub["rl_p50"] - sub["ttft"]) / 1000.0)
        tot = pfj + dcj
        ratio = (pfj / tot).median() * 100 if tot.notna().any() else np.nan
        prefill_pct.append(ratio)
        decode_pct.append(100 - ratio if not np.isnan(ratio) else np.nan)
    return prefill_pct, decode_pct

fig, ax = plt.subplots(figsize=(11, 6.5))
w = 0.34
prefill_pct_lc, decode_pct_lc = energy_split_pct(df_lc, MODELS)
prefill_pct_mlx_3, decode_pct_mlx_3 = energy_split_pct(df_mlx, MLXLM_MODELS)
mlx_idx = {m: i for i, m in enumerate(MLXLM_MODELS)}
prefill_pct_mlx = [prefill_pct_mlx_3[mlx_idx[m]] if m in mlx_idx else np.nan for m in MODELS]
decode_pct_mlx  = [decode_pct_mlx_3[mlx_idx[m]] if m in mlx_idx else np.nan for m in MODELS]

bars_p_lc = ax.bar(x - w / 2, prefill_pct_lc, w, color="#FF9800", edgecolor="white", label="Prefill % (llama.cpp)")
bars_d_lc = ax.bar(x - w / 2, decode_pct_lc, w, bottom=prefill_pct_lc, color="#2196F3", edgecolor="white", label="Decode % (llama.cpp)")
bars_p_mx = ax.bar(x + w / 2, prefill_pct_mlx, w, color="#FF9800", edgecolor="black", hatch="//", alpha=0.85, label="Prefill % (MLX-LM)")
bars_d_mx = ax.bar(x + w / 2, decode_pct_mlx, w, bottom=prefill_pct_mlx, color="#2196F3", edgecolor="black", hatch="//", alpha=0.85, label="Decode % (MLX-LM)")
for i in range(len(MODELS)):
    if not np.isnan(prefill_pct_lc[i]):
        ax.text(x[i] - w / 2, prefill_pct_lc[i] / 2, f"{prefill_pct_lc[i]:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x[i] - w / 2, prefill_pct_lc[i] + decode_pct_lc[i] / 2, f"{decode_pct_lc[i]:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    if not np.isnan(prefill_pct_mlx[i]):
        ax.text(x[i] + w / 2, prefill_pct_mlx[i] / 2, f"{prefill_pct_mlx[i]:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x[i] + w / 2, prefill_pct_mlx[i] + decode_pct_mlx[i] / 2, f"{decode_pct_mlx[i]:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([MDL_FLAT[m] for m in MODELS], rotation=15, ha="right", fontsize=9)
ax.set_title("Request Energy Split: Prefill % vs Decode % (median across all combos) — llama.cpp vs MLX-LM", fontweight="bold", fontsize=12)
ax.set_ylabel("% of total request energy")
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, 115)
plt.tight_layout()
save(fig, "E_prefill_decode_energy_split.png")

# Context-length retention: tok/s at ctx=30720 (longest tested) vs ctx=256 (gen=1024), both backends
LONGEST_CTX = max(PROMPT_LENGTHS)
def retention_for(df_sub, models):
    out = {}
    for m in models:
        s256 = df_sub[(df_sub.model == m) & (df_sub.gen == CANONICAL_GEN) & (df_sub.prompt == 256)]["tok_s"]
        slong = df_sub[(df_sub.model == m) & (df_sub.gen == CANONICAL_GEN) & (df_sub.prompt == LONGEST_CTX)]["tok_s"]
        if not s256.empty and not slong.empty:
            out[m] = float(slong.iloc[0]) / float(s256.iloc[0]) * 100
    return out

retention_lc  = retention_for(df_lc, MODELS)
retention_mlx = retention_for(df_mlx, MLXLM_MODELS)
bar_chart_dual(retention_lc, retention_mlx, "Tok/s retained (%)",
               f"Output Tok/s Retained at ctx={LONGEST_CTX} vs ctx=256 (gen=1024) — llama.cpp vs MLX-LM",
               "context_length_retention_bar.png", fmt="{:.1f}%")


# ═════════════════════════════ Heatmaps ═════════════════════════════
# A single heatmap cell can only show one number, so a per-cell backend overlay
# isn't possible the way it is for line/bar charts. Instead: the primary 2x3
# grid stays llama.cpp (all 6 models — the complete dataset), and a companion
# 1x3 MLX-LM grid (the 3 shared models) is generated right after it for every
# metric, so both backends are still available for every heatmapped metric —
# just as two adjacent figures instead of one merged one.
def heatmap_grid(df_src, models, nrows, ncols, metric, label, fmt_str, cmap, fname,
                  vmin=None, vmax=None, backend_label=""):
    fig, axes = plt.subplots(nrows, ncols, figsize=(24 if ncols == 3 and nrows == 2 else 8 * ncols, 11 if nrows == 2 else 5.5))
    axes = np.atleast_1d(axes).flatten()
    suffix = f" — {backend_label}" if backend_label else ""
    fig.suptitle(f"{label} — All Prompt x Gen Combinations, All Models{suffix}", fontsize=16, fontweight="bold")
    for idx, model in enumerate(models):
        ax = axes[idx]
        sub = df_src[df_src.model == model]
        if sub.empty or sub[metric].isna().all():
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        pivot = sub.pivot_table(index="gen", columns="prompt", values=metric, aggfunc="median")
        pivot.columns = [PROMPT_LABEL_MAP.get(int(c), str(c)) for c in pivot.columns]
        sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt_str, cmap=cmap,
                    linewidths=1, cbar=True, vmin=vmin, vmax=vmax,
                    annot_kws={"fontsize": 13, "fontweight": "bold"},
                    cbar_kws={"shrink": 0.85})
        ax.tick_params(axis="both", labelsize=12)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=11)
        ax.set_title(MDL_FLAT[model], fontsize=14, fontweight="bold")
        ax.set_xlabel("Prompt (tok)" if idx >= (ncols if nrows > 1 else 0) else "", fontsize=12)
        ax.set_ylabel("Gen (tok)" if idx % ncols == 0 else "", fontsize=12)
    plt.tight_layout()
    save(fig, fname)


print("\n-- Heatmaps: llama.cpp (all 6 models) --")
tokj_all = df_lc["tok_j"].dropna()
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "tok_j", "Output Tok/J", ".2f", "YlGnBu",
             "7_tok_j_heatmap_all_models.png", vmin=tokj_all.min(), vmax=tokj_all.max(),
             backend_label="llama.cpp")
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "tok_s", "Output Tok/s", ".1f", "Blues",
             "E_tok_s_heatmap_all_models.png", backend_label="llama.cpp")
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "itl", "ITL p50 (ms)", ".1f", "Oranges",
             "EH_itl_heatmap_all_models.png", backend_label="llama.cpp")
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "ttft", "TTFT p50 (ms)", ".0f", "Reds",
             "EG_ttft_heatmap_all_models.png", backend_label="llama.cpp")
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "prefill", "Prefill Tok/s", ".0f", "Greens",
             "EI_prefill_tput_heatmap_all_models.png", backend_label="llama.cpp")
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "prefill_pw", "Prefill Power (W)", ".2f", "OrRd",
             "EP_prefill_power_heatmap_all_models.png", backend_label="llama.cpp")
heatmap_grid(df_lc, MODELS, NROWS, NCOLS, "decode_pw", "Decode Power (W)", ".2f", "PuBu",
             "EP_decode_power_heatmap_all_models.png", backend_label="llama.cpp")

print("\n-- Heatmaps: MLX-LM (3 shared models) --")
tokj_mlx_all = df_mlx["tok_j"].dropna()
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "tok_j", "Output Tok/J", ".2f", "YlGnBu",
             "7_tok_j_heatmap_mlxlm.png",
             vmin=tokj_mlx_all.min() if not tokj_mlx_all.empty else None,
             vmax=tokj_mlx_all.max() if not tokj_mlx_all.empty else None, backend_label="MLX-LM")
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "tok_s", "Output Tok/s", ".1f", "Blues",
             "E_tok_s_heatmap_mlxlm.png", backend_label="MLX-LM")
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "itl", "ITL p50 (ms)", ".1f", "Oranges",
             "EH_itl_heatmap_mlxlm.png", backend_label="MLX-LM")
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "ttft", "TTFT p50 (ms)", ".0f", "Reds",
             "EG_ttft_heatmap_mlxlm.png", backend_label="MLX-LM")
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "prefill", "Prefill Tok/s", ".0f", "Greens",
             "EI_prefill_tput_heatmap_mlxlm.png", backend_label="MLX-LM")
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "prefill_pw", "Prefill Power (W)", ".2f", "OrRd",
             "EP_prefill_power_heatmap_mlxlm.png", backend_label="MLX-LM")
heatmap_grid(df_mlx, MLXLM_MODELS, 1, 3, "decode_pw", "Decode Power (W)", ".2f", "PuBu",
             "EP_decode_power_heatmap_mlxlm.png", backend_label="MLX-LM")

print(f"\nAll charts saved to {OUT_DIR}")
