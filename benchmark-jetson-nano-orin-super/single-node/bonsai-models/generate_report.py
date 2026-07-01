#!/usr/bin/env python3
"""
generate_report.py — Bonsai Benchmark Per-Run Report Generator

Usage:
    python3 generate_report.py artifacts/llamacpp/bonsai-llamacpp-YYYYMMDD-HHMM/

Reads profile_export_aiperf.json + profile_export.jsonl per cell,
tegrastats.log and model_timing.log from the artifact root.
Writes RESULTS.md in the artifact directory.

tok/J uses exact per-request phase power (same method as generate_combined_charts.py):
  decode_j  = decode_power_W × p50_decode_s
  output_tok_j = OSL / decode_j
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np


# ── Tegrastats loader ─────────────────────────────────────────────────────────

def load_tegra(base_dir):
    records = []
    path = os.path.join(base_dir, "tegrastats.log")
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
            if not m:
                continue
            try:
                ep = datetime.strptime(m.group(1), '%m-%d-%Y %H:%M:%S').timestamp()
            except ValueError:
                continue
            pw = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
            tj = re.search(r'tj@([\d.]+)C', line)
            if pw:
                records.append((ep, int(pw.group(1)), float(tj.group(1)) if tj else None))
    return records


def load_model_windows(base_dir):
    windows = {}
    path = os.path.join(base_dir, "model_timing.log")
    if not os.path.exists(path):
        return windows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MODEL_START:"):
                _, name, ts = line.split(":", 2)
                windows.setdefault(name, {})["start"] = float(ts)
            elif line.startswith("MODEL_END:"):
                _, name, ts = line.split(":", 2)
                windows.setdefault(name, {})["end"] = float(ts)
    return windows


# ── Phase-separated power from jsonl ─────────────────────────────────────────

def compute_tok_j(aiperf_path, tegra_records):
    """Return (tok_j, prefill_power_w, decode_power_w, decode_j) or Nones on failure."""
    try:
        d = json.load(open(aiperf_path))
    except Exception:
        return None, None, None, None

    def pct(k, v="p50"):
        return (d.get(k) or {}).get(v)

    osl    = pct("output_sequence_length", "p50")
    ttft   = pct("time_to_first_token",    "p50")
    t0_str = d.get("start_time")
    t1_str = d.get("end_time")

    if not (osl and ttft and t0_str and t1_str):
        return None, None, None, None

    t0 = datetime.fromisoformat(t0_str).timestamp()
    t1 = datetime.fromisoformat(t1_str).timestamp()
    samp_records = [(ep, mw) for (ep, mw, _) in tegra_records if t0 <= ep <= t1]
    if not samp_records:
        return None, None, None, None

    jsonl_path = os.path.join(os.path.dirname(aiperf_path), "profile_export.jsonl")
    prefill_mw, decode_mw = [], []
    p50_decode_s = None

    if os.path.exists(jsonl_path):
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

            for ep, mw in samp_records:
                if any(ws <= ep <= wa for ws, wa in prefill_wins):
                    prefill_mw.append(mw)
                elif any(wa < ep <= we for wa, we in decode_wins):
                    decode_mw.append(mw)

            p50_decode_s = float(np.median([(e - a) / 1e9 for s, a, e in per_req]))

    # Fallback to timeline reconstruction if jsonl missing or no samples classified
    if not prefill_mw or not decode_mw:
        rl_p50_ms = pct("request_latency", "p50")
        if rl_p50_ms and ttft:
            rl_s   = rl_p50_ms / 1000.0
            ttft_s = ttft / 1000.0
            n_reqs = int(pct("request_count", "avg") or 20)
            for ep, mw in samp_records:
                elapsed      = ep - t0
                req_idx      = int(elapsed / rl_s)
                if req_idx >= n_reqs:
                    continue
                phase_elapsed = elapsed - req_idx * rl_s
                if phase_elapsed <= ttft_s:
                    prefill_mw.append(mw)
                else:
                    decode_mw.append(mw)
            if p50_decode_s is None:
                p50_decode_s = rl_s - ttft / 1000.0

    all_mw = [mw for ep, mw in samp_records]
    fallback_w = float(np.median(all_mw)) / 1000 if all_mw else None

    prefill_power_w = float(np.median(prefill_mw)) / 1000 if prefill_mw else fallback_w
    decode_power_w  = float(np.median(decode_mw))  / 1000 if decode_mw  else fallback_w

    p50_ttft_s = ttft / 1000.0
    decode_j   = decode_power_w  * p50_decode_s if (decode_power_w  and p50_decode_s) else None
    tok_j      = osl / decode_j  if (osl and decode_j and decode_j > 0) else None

    return tok_j, prefill_power_w, decode_power_w, decode_j


# ── Results loader ─────────────────────────────────────────────────────────────

def load_results(base_dir, tegra_records):
    results = []
    pattern = os.path.join(base_dir, "**", "profile_export_aiperf.json")
    for path in glob.glob(pattern, recursive=True):
        rel   = os.path.relpath(path, base_dir)
        parts = rel.split(os.sep)
        # Expected: {ModelName}/gen{G}/ctx{C}/profile_export_aiperf.json
        if len(parts) < 4:
            continue
        model_name = parts[0]
        gen_m = re.search(r'(\d+)', parts[1])
        ctx_m = re.search(r'(\d+)', parts[2])
        if not gen_m or not ctx_m:
            continue
        gen = int(gen_m.group(1))
        ctx = int(ctx_m.group(1))

        try:
            d = json.load(open(path))
        except Exception:
            continue

        def pct(k, v="p50"):
            return (d.get(k) or {}).get(v)

        ttft    = pct("time_to_first_token",              "p50")
        itl     = pct("inter_token_latency",              "p50")
        tok_s   = pct("output_token_throughput_per_user", "p50")
        rl_p50  = pct("request_latency",                  "p50")
        osl     = pct("output_sequence_length",           "p50")
        prefill = pct("prefill_throughput_per_user",      "p50")
        err_cnt = (d.get("error_request_count") or {}).get("avg")

        tok_j, prefill_pw, decode_pw, decode_j = compute_tok_j(path, tegra_records)

        quant = "Q2_0 (1.58-bit)" if "Ternary" in model_name else "Q1_0 (1-bit)"

        results.append({
            "model":          model_name,
            "quant":          quant,
            "ctx":            ctx,
            "gen":            gen,
            "ttft_ms":        round(ttft,    1) if ttft    else None,
            "itl_ms":         round(itl,     3) if itl     else None,
            "tok_s":          round(tok_s,   2) if tok_s   else None,
            "rl_ms":          round(rl_p50,  1) if rl_p50  else None,
            "osl":            round(osl,     1) if osl     else None,
            "prefill_tps":    round(prefill, 1) if prefill else None,
            "tok_j":          round(tok_j,   4) if tok_j   else None,
            "prefill_pw":     round(prefill_pw, 3) if prefill_pw else None,
            "decode_pw":      round(decode_pw,  3) if decode_pw  else None,
            "decode_j":       round(decode_j,   5) if decode_j   else None,
            "errors":         int(err_cnt) if err_cnt else 0,
            "path":           path,
        })

    return sorted(results, key=lambda r: (r["model"], r["ctx"], r["gen"]))


# ── Thermal summary from tegrastats ──────────────────────────────────────────

def thermal_summary(tegra_records, model_windows):
    summary = {}
    for model, win in model_windows.items():
        t0, t1 = win.get("start", 0), win.get("end", 9e18)
        w = [(mw, tj) for (ep, mw, tj) in tegra_records if t0 <= ep <= t1]
        if not w:
            continue
        mw_vals = [x[0] for x in w]
        tj_vals  = [x[1] for x in w if x[1] is not None]
        summary[model] = {
            "avg_pw": round(float(np.median(mw_vals)) / 1000, 2),
            "peak_tj": round(max(tj_vals), 1) if tj_vals else None,
            "throttled": max(tj_vals) > 85 if tj_vals else False,
        }
    return summary


# ── Markdown table helper ─────────────────────────────────────────────────────

def md_table(headers, rows, alignments=None):
    if not rows:
        return "*No data.*\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep_parts = []
    for i, w in enumerate(widths):
        align = alignments[i] if alignments else "left"
        if align == "right":
            sep_parts.append("-" * (w - 1) + ":")
        elif align == "center":
            sep_parts.append(":" + "-" * (w - 2) + ":")
        else:
            sep_parts.append("-" * w)
    return "\n".join([
        fmt_row(headers),
        "| " + " | ".join(s.ljust(widths[i]) for i, s in enumerate(sep_parts)) + " |",
        *[fmt_row(r) for r in rows],
    ]) + "\n"


def fmt(v, spec, fallback="-"):
    return format(v, spec) if v is not None else fallback


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from bonsai benchmark artifacts")
    parser.add_argument("artifact_dir", help="Path to bonsai artifact dir (e.g. artifacts/llamacpp/bonsai-llamacpp-YYYYMMDD-HHMM/)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.artifact_dir)
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Reading artifacts from: {base_dir}")
    tegra_records  = load_tegra(base_dir)
    model_windows  = load_model_windows(base_dir)
    results        = load_results(base_dir, tegra_records)
    therm          = thermal_summary(tegra_records, model_windows)
    print(f"  {len(tegra_records)} tegrastats samples  |  {len(results)} result cells")

    sections = []

    # Header
    sections.append(
        f"# Bonsai Benchmark Results — Jetson Orin Nano Super 8GB\n\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  \n"
        f"*Artifact: `{os.path.basename(base_dir)}`*\n"
    )

    # Primary results table (all cells)
    r_aligns = ["left", "left", "right", "right", "right", "right", "right", "right", "right", "right"]
    r_hdrs   = ["Model", "Quant", "Ctx", "Gen", "TTFT p50 (ms)", "ITL p50 (ms)",
                "Tok/s p50", "Decode W", "Tok/J", "Errors"]
    r_rows = []
    for r in results:
        r_rows.append([
            r["model"], r["quant"],
            r["ctx"], r["gen"],
            fmt(r["ttft_ms"], ".1f"),
            fmt(r["itl_ms"],  ".2f"),
            fmt(r["tok_s"],   ".2f"),
            fmt(r["decode_pw"], ".2f"),
            fmt(r["tok_j"],   ".4f"),
            r["errors"] or "-",
        ])
    sections.append("## Full Results\n\n" + md_table(r_hdrs, r_rows, r_aligns))

    # Best tok/J per model
    best = {}
    for r in results:
        m = r["model"]
        if r["tok_j"] is not None:
            if m not in best or r["tok_j"] > best[m]["tok_j"]:
                best[m] = r
    if best:
        b_hdrs   = ["Model", "Best Tok/J", "Ctx", "Gen", "Tok/s", "Prefill W", "Decode W"]
        b_aligns = ["left", "right", "right", "right", "right", "right", "right"]
        b_rows   = [
            [r["model"], fmt(r["tok_j"], ".4f"), r["ctx"], r["gen"],
             fmt(r["tok_s"], ".2f"), fmt(r["prefill_pw"], ".2f"), fmt(r["decode_pw"], ".2f")]
            for r in sorted(best.values(), key=lambda x: x["tok_j"], reverse=True)
        ]
        sections.append("## Best Tok/J per Model\n\n" + md_table(b_hdrs, b_rows, b_aligns))

    # Thermal summary
    if therm:
        t_hdrs = ["Model", "Avg Power (W)", "Peak TJ (C)", "Throttled"]
        t_rows = [
            [m, fmt(v["avg_pw"], ".2f"), fmt(v["peak_tj"], ".1f"),
             "YES" if v["throttled"] else "No"]
            for m, v in sorted(therm.items())
        ]
        sections.append("## Thermal Summary\n\n" + md_table(t_hdrs, t_rows))

    # Errors
    errors = [r for r in results if r["errors"] > 0]
    if errors:
        e_hdrs = ["Model", "Ctx", "Gen", "Errors"]
        e_rows = [[r["model"], r["ctx"], r["gen"], r["errors"]] for r in errors]
        sections.append("## Failed Cells\n\n" + md_table(e_hdrs, e_rows))
    else:
        sections.append("## Failed Cells\n\n*No failed cells.*\n")

    # Methodology
    sections.append(
        "## Methodology\n\n"
        "- **Power rail**: `VDD_CPU_GPU_CV` from tegrastats at 500ms interval\n"
        "- **Phase separation**: exact per-request prefill/decode windows from `profile_export.jsonl` "
        "(`request_start_ns`, `request_ack_ns`, `request_end_ns`)\n"
        "- **Decode power**: median of tegrastats samples falling inside decode windows\n"
        "- **Tok/J**: `OSL / (decode_power_W × p50_decode_s)` — output tokens per joule of decode energy\n"
        "- **Prefill power**: median of tegrastats samples falling inside prefill windows\n"
        "- **Fallback**: timeline reconstruction used if `profile_export.jsonl` is absent\n"
        "- **Concurrency**: 1\n"
        "- **Clock locking**: `jetson_clocks` (max clocks)\n"
    )

    out_path = os.path.join(base_dir, "RESULTS.md")
    with open(out_path, "w") as f:
        f.write("\n".join(sections))

    print(f"\nReport written: {out_path}")
    print(f"  {len(results)} cells  |  {len(best)} models  |  {len(errors)} failed cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
