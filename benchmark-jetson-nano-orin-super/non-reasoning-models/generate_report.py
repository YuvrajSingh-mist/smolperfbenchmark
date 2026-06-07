#!/usr/bin/env python3
"""
generate_report.py — Jetson Orin LLM Benchmark Report Generator

Usage:
    python3 generate_report.py artifacts/blog-20260524-1200/

Reads all profile_export_aiperf.json files, FAILURES.log, model-exam-*.log,
thermal_summary.log; writes RESULTS.md in the artifact directory.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime


def load_results(base_dir):
    results = []
    pattern = os.path.join(base_dir, "**", "profile_export_aiperf.json")
    for path in glob.glob(pattern, recursive=True):
        rel = os.path.relpath(path, base_dir)
        parts = rel.split(os.sep)
        # Expected: {backend}/{model-safe-quant}/ctx-{C}_out-{O}/profile_export_aiperf.json
        if len(parts) < 4:
            continue
        backend = parts[0]
        if backend not in ("ollama", "llamacpp"):
            continue
        model_quant_safe = parts[1]
        cell_dir = parts[2]

        ctx_m = re.search(r"ctx-(\d+)", cell_dir)
        out_m = re.search(r"out-(\d+)", cell_dir)
        if not ctx_m or not out_m:
            continue
        ctx = int(ctx_m.group(1))
        out_len = int(out_m.group(1))

        # Split model_quant_safe: model-safe + quant suffix
        # quant is lowercase, e.g. q8_0 / q4_k_m
        quant_m = re.search(r"-([qQ]\d+[_k]*[_m]*)$", model_quant_safe)
        if quant_m:
            quant = quant_m.group(1).upper()
            model_safe = model_quant_safe[: quant_m.start()]
        else:
            quant = "unknown"
            model_safe = model_quant_safe

        # Reconstruct display model name (best effort)
        model_display = model_safe.replace("-", ":", 1)

        try:
            d = json.load(open(path))
            ttft = (d.get("time_to_first_token", {}) or {}).get("avg", 0)
            itl = (d.get("inter_token_latency", {}) or {}).get("avg", 0)
            tps = (d.get("output_token_throughput_per_user", {}) or {}).get("avg", 0)
            rl = (d.get("request_latency", {}) or {}).get("avg", 0)
            results.append(
                {
                    "backend": backend,
                    "model_safe": model_safe,
                    "model": model_display,
                    "quant": quant,
                    "ctx": ctx,
                    "out_len": out_len,
                    "ttft_ms": round(ttft, 1),
                    "itl_ms": round(itl, 3),
                    "tps": round(tps, 2),
                    "req_lat_s": round(rl / 1000, 2),
                    "path": path,
                }
            )
        except Exception as e:
            print(f"  [!] Error reading {path}: {e}", file=sys.stderr)

    return results


def load_failures(base_dir):
    failures = []
    flog = os.path.join(base_dir, "FAILURES.log")
    if not os.path.exists(flog):
        return failures
    with open(flog) as f:
        for line in f:
            line = line.strip()
            if not line or "FAIL" not in line:
                continue
            # Format: [ts] FAIL  backend=X  model=Y  reason=Z
            ts_m = re.match(r"\[([^\]]+)\]", line)
            ts = ts_m.group(1) if ts_m else ""
            backend = re.search(r"backend=(\S+)", line)
            model = re.search(r"model=(\S+)", line)
            reason = re.search(r"reason=(.+)$", line)
            failures.append(
                {
                    "ts": ts,
                    "backend": backend.group(1) if backend else "",
                    "model": model.group(1) if model else "",
                    "reason": reason.group(1) if reason else line,
                }
            )
    return failures


def load_exam_log(base_dir):
    exam_entries = []
    pattern = os.path.join(base_dir, "model-exam-*.log")
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("PASS") or line.startswith("FAIL"):
                    exam_entries.append(line)
    return exam_entries


def load_thermal(base_dir):
    thermal = {"avg_power_w": None, "max_tj_c": None, "throttled": False}
    tlog = os.path.join(base_dir, "thermal_summary.log")
    if not os.path.exists(tlog):
        return thermal
    content = open(tlog).read()
    # Parse avg power after cutoff
    m = re.search(r"Power \(after cutoff\).*?avg=([\d.]+)W", content)
    if m:
        thermal["avg_power_w"] = float(m.group(1))
    # Max TJ temp
    m = re.search(r"TJ.*?max=([\d.]+)C", content)
    if m:
        thermal["max_tj_c"] = float(m.group(1))
    thermal["throttled"] = "THROTTLED" in content
    return thermal


def md_table(headers, rows, alignments=None):
    if not rows:
        return "*No data.*\n"
    # Determine column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = []
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines.append(header_line)
    sep_parts = []
    for i, w in enumerate(widths):
        align = alignments[i] if alignments else "left"
        if align == "right":
            sep_parts.append("-" * (w - 1) + ":")
        elif align == "center":
            sep_parts.append(":" + "-" * (w - 2) + ":")
        else:
            sep_parts.append("-" * w)
    lines.append("| " + " | ".join(s.ljust(widths[i]) for i, s in enumerate(sep_parts)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines) + "\n"


def get_system_info():
    info = {}
    # JetPack version
    try:
        v = open("/etc/nv_tegra_release").readline().strip()
        info["jetpack"] = v
    except Exception:
        info["jetpack"] = "unknown"
    # Ollama version
    try:
        import subprocess
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        info["ollama"] = r.stdout.strip() or r.stderr.strip()
    except Exception:
        info["ollama"] = "unknown"
    # llama.cpp git SHA
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", os.path.expanduser("~/llama.cpp"), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        info["llamacpp_sha"] = r.stdout.strip()
    except Exception:
        info["llamacpp_sha"] = "unknown"
    return info


def main():
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from benchmark artifacts")
    parser.add_argument("artifact_dir", help="Path to artifacts/blog-DATE/ directory")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.artifact_dir)
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Reading artifacts from: {base_dir}")
    results = load_results(base_dir)
    failures = load_failures(base_dir)
    exam_entries = load_exam_log(base_dir)
    thermal = load_thermal(base_dir)
    sysinfo = get_system_info()

    print(f"  Found {len(results)} result cells, {len(failures)} failures, {len(exam_entries)} exam entries")

    # Compute tok/J
    avg_power = thermal.get("avg_power_w")
    for r in results:
        if avg_power:
            r["tok_per_j"] = round(r["tps"] / avg_power, 4)
        else:
            r["tok_per_j"] = None

    out_path = os.path.join(base_dir, "RESULTS.md")
    sections = []

    # ── Header ──────────────────────────────────────────────────────────────────
    sections.append(
        f"# Jetson Nano Super Orin 8GB — LLM Efficiency Benchmark\n\n"
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    )

    # ── System info ─────────────────────────────────────────────────────────────
    sys_rows = [
        ["Device", "NVIDIA Jetson Orin Nano Super 8GB"],
        ["JetPack", sysinfo.get("jetpack", "unknown")],
        ["Ollama", sysinfo.get("ollama", "unknown")],
        ["llama.cpp", f"git {sysinfo.get('llamacpp_sha', 'unknown')}"],
        ["Run date", datetime.now().strftime("%Y-%m-%d")],
        ["Models validated", f"{sum(1 for e in exam_entries if e.startswith('PASS'))} passed, "
                              f"{sum(1 for e in exam_entries if e.startswith('FAIL'))} dropped"],
    ]
    sections.append("## System\n\n" + md_table(["Field", "Value"], sys_rows))

    # ── Model exam results ───────────────────────────────────────────────────────
    if exam_entries:
        exam_rows = []
        for entry in exam_entries:
            status = "✓ PASS" if entry.startswith("PASS") else "✗ FAIL"
            parts = entry.split(maxsplit=2)
            model = parts[1] if len(parts) > 1 else entry
            reason = parts[2] if len(parts) > 2 else "—"
            exam_rows.append([model, status, reason])
        sections.append("## Model Validation Exam\n\n" + md_table(["Model", "Result", "Details"], exam_rows))
    else:
        sections.append("## Model Validation Exam\n\n*No exam log found.*\n")

    # ── Primary results (ctx=512, out=128) ───────────────────────────────────────
    primary = [r for r in results if r["ctx"] == 512 and r["out_len"] == 128]
    primary.sort(key=lambda x: x["tps"], reverse=True)

    if primary:
        prows = []
        for r in primary:
            prows.append([
                r["model"], r["backend"], r["quant"],
                f"{r['tps']:.2f}", f"{r['ttft_ms']:.0f}", f"{r['itl_ms']:.2f}",
                f"{avg_power:.1f}" if avg_power else "—",
                f"{r['tok_per_j']:.4f}" if r["tok_per_j"] else "—",
                f"{thermal['max_tj_c']:.1f}" if thermal["max_tj_c"] else "—",
            ])
        hdrs = ["Model", "Backend", "Quant", "tok/s ↑", "TTFT ms", "ITL ms", "Avg W", "tok/J ↑", "Max °C"]
        aligns = ["left", "left", "left", "right", "right", "right", "right", "right", "right"]
        sections.append("## Primary Results — ctx=512, out=128\n\n" + md_table(hdrs, prows, aligns))
    else:
        sections.append("## Primary Results — ctx=512, out=128\n\n*No results at ctx=512/out=128.*\n")

    # ── Context scaling table (out=128 fixed) ───────────────────────────────────
    ctx_vals = sorted(set(r["ctx"] for r in results if r["out_len"] == 128))
    if len(ctx_vals) > 1:
        # Get unique (model, backend, quant) combos
        combos = sorted(set((r["model"], r["backend"], r["quant"]) for r in results if r["out_len"] == 128))
        ctx_hdrs = ["Model", "Backend", "Quant"] + [f"ctx={c}" for c in ctx_vals]
        ctx_rows = []
        for model, backend, quant in combos:
            row = [model, backend, quant]
            for c in ctx_vals:
                match = next(
                    (r for r in results if r["model"] == model and r["backend"] == backend
                     and r["quant"] == quant and r["ctx"] == c and r["out_len"] == 128), None
                )
                row.append(f"{match['tps']:.2f}" if match else "—")
            ctx_rows.append(row)
        sections.append("## Context Scaling — tok/s (out=128 fixed)\n\n" + md_table(ctx_hdrs, ctx_rows))
    else:
        sections.append("## Context Scaling\n\n*Only one context length measured.*\n")

    # ── Output length scaling (ctx=512 fixed) ───────────────────────────────────
    out_vals = sorted(set(r["out_len"] for r in results if r["ctx"] == 512))
    if len(out_vals) > 1:
        combos = sorted(set((r["model"], r["backend"], r["quant"]) for r in results if r["ctx"] == 512))
        out_hdrs = ["Model", "Backend", "Quant"] + [f"out={o}" for o in out_vals]
        out_rows = []
        for model, backend, quant in combos:
            row = [model, backend, quant]
            for o in out_vals:
                match = next(
                    (r for r in results if r["model"] == model and r["backend"] == backend
                     and r["quant"] == quant and r["ctx"] == 512 and r["out_len"] == o), None
                )
                row.append(f"{match['tps']:.2f}" if match else "—")
            out_rows.append(row)
        sections.append("## Output Length Scaling — tok/s (ctx=512 fixed)\n\n" + md_table(out_hdrs, out_rows))
    else:
        sections.append("## Output Length Scaling\n\n*Only one output length measured.*\n")

    # ── tok/J efficiency ranking (ctx=512, out=128) ──────────────────────────────
    eff = [r for r in primary if r["tok_per_j"] is not None]
    eff.sort(key=lambda x: x["tok_per_j"], reverse=True)
    if eff:
        eff_rows = [
            [str(i + 1), r["model"], r["backend"], r["quant"], f"{r['tps']:.2f}",
             f"{avg_power:.1f}" if avg_power else "—", f"{r['tok_per_j']:.4f}"]
            for i, r in enumerate(eff)
        ]
        eff_hdrs = ["Rank", "Model", "Backend", "Quant", "tok/s", "Avg W", "tok/J ↑"]
        sections.append("## Efficiency Ranking — tok/J (ctx=512, out=128)\n\n" + md_table(eff_hdrs, eff_rows))
    else:
        sections.append("## Efficiency Ranking\n\n*Insufficient data (power readings or results missing).*\n")

    # ── Failed runs ──────────────────────────────────────────────────────────────
    if failures:
        fail_rows = [[f["ts"], f["model"], f["backend"], f["reason"]] for f in failures]
        sections.append("## Failed Runs\n\n" + md_table(["Timestamp", "Model", "Backend", "Reason"], fail_rows))
    else:
        sections.append("## Failed Runs\n\n*No failures recorded.*\n")

    # ── Thermal summary ──────────────────────────────────────────────────────────
    therm_rows = [
        ["Avg power (after startup)", f"{avg_power:.2f} W" if avg_power else "—"],
        ["Max TJ temp", f"{thermal['max_tj_c']:.1f} °C" if thermal["max_tj_c"] else "—"],
        ["Throttled", "YES ⚠" if thermal["throttled"] else "No"],
    ]
    tlog_content = ""
    tlog_path = os.path.join(base_dir, "thermal_summary.log")
    if os.path.exists(tlog_path):
        tlog_content = "\n```\n" + open(tlog_path).read() + "\n```\n"
    sections.append("## Thermal Summary\n\n" + md_table(["Metric", "Value"], therm_rows) + tlog_content)

    # ── Methodology ──────────────────────────────────────────────────────────────
    ctx_list_str = ", ".join(str(c) for c in sorted(set(r["ctx"] for r in results))) or "—"
    out_list_str = ", ".join(str(o) for o in sorted(set(r["out_len"] for r in results))) or "—"
    sections.append(
        f"""## Methodology

- **Device**: NVIDIA Jetson Orin Nano Super 8GB
- **Clock locking**: `nvpmodel -m 0` + `jetson_clocks` (max performance mode)
- **Concurrency**: 1 (single-user latency characterisation)
- **Requests per cell**: {results[0]['path'] and '(see run config)' if results else '—'}
- **Context sweep**: {ctx_list_str} tokens
- **Output sweep**: {out_list_str} tokens
- **Tool**: [aiperf](https://github.com/triton-inference-server/perf_analyzer) for both backends
- **Thinking**: disabled (`think: false`) for all models — non-thinking benchmark
- **GPU**: mandatory for all models — any model not confirmed on GPU is excluded
- **Quantization**: matched exactly between Ollama and llama.cpp (GGUF downloaded to match)
- **Ollama streaming**: disabled for models with `think=false` (workaround for Ollama 0.24.0 bug)
- **llama.cpp flags**: `--n-gpu-layers 999` (full GPU offload), `--ctx-size 8192`
- **Power metric**: `VDD_CPU_GPU_CV` from tegrastats @ 500ms interval; startup TTFT dropped
- **tok/J**: `output_token_throughput_per_user (avg) / avg_power_W`
- **Random seed**: 42
"""
    )

    # Write output
    with open(out_path, "w") as f:
        f.write("\n".join(sections))

    print(f"\nReport written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
