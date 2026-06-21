#!/usr/bin/env python3
"""
generate_report.py — Android smolbench-mobile
Mirrors the Jetson report format: per-cell table + thermal summary + per-model best.
Reads:
  - aiperf profile_export_aiperf.json  (TTFT, ITL, tok/s, request latency)
  - thermal.log  (CSV: timestamp_ms, zone1_mdegC, zone2_mdegC, ...)
  - model_timing.log  (MODEL_START/END markers)
Output:
  - report.md in base_dir
"""
import sys
import json
import os
import glob
import re
from datetime import datetime
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
base_dir      = sys.argv[1] if len(sys.argv) > 1 else "."
thermal_log   = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base_dir, "thermal.log")
timing_log    = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_dir, "model_timing.log")
skipped_arg   = sys.argv[4] if len(sys.argv) > 4 else ""
ctx_size      = sys.argv[5] if len(sys.argv) > 5 else "?"
framework     = sys.argv[6] if len(sys.argv) > 6 else "llamacpp"
device_name   = sys.argv[7] if len(sys.argv) > 7 else "unknown"

cpufreq_log   = os.path.join(base_dir, "cpufreq.log")
battery_log   = os.path.join(base_dir, "battery.log")

report_path   = os.path.join(base_dir, "report.md")
skipped       = [s for s in skipped_arg.split("||") if s] if skipped_arg else []


# ── Thermal log parsing ───────────────────────────────────────────────────────
# thermal.log format:
# Line 0 (header): timestamp_ms,cpu_cpu-0-0_mdegC,gpu_gpu_mdegC,...
# Lines 1+:        1718500000000,62000,55000,...
# ITER_START/END markers are interspersed for alignment.

class ThermalFrame:
    def __init__(self, ts_ms: float, vals: dict[str, float]):
        self.ts_ms = ts_ms      # epoch ms
        self.vals  = vals       # {zone_key: temp_millidegC}


thermal_frames: list[ThermalFrame] = []
thermal_col_names: list[str] = []
iter_markers: list[dict] = []   # {event, model, gen, ctx, ts_ms}

try:
    with open(thermal_log) as f:
        header_line = f.readline().strip()
        # Header: timestamp_ms,col1,col2,...
        all_cols = header_line.split(",")
        thermal_col_names = all_cols[1:]   # skip timestamp_ms

        for raw in f:
            raw = raw.strip()
            if not raw:
                continue

            # ITER markers embedded in log
            if raw.startswith("ITER_"):
                parts = raw.split(":")
                if len(parts) >= 5:
                    event   = parts[0]      # ITER_START or ITER_END
                    model   = parts[1]
                    gen_s   = parts[2]      # gen128
                    ctx_s   = parts[3]      # ctx256
                    ts_m    = parts[4]
                    try:
                        iter_markers.append({
                            "event":  event,
                            "model":  model,
                            "gen":    int(re.sub(r"\D", "", gen_s)),
                            "ctx":    int(re.sub(r"\D", "", ctx_s)),
                            "ts_ms":  float(ts_m),
                        })
                    except ValueError:
                        pass
                continue

            # Data row
            cols = raw.split(",")
            if len(cols) < 2:
                continue
            try:
                ts_ms = float(cols[0])
                vals = {}
                for j, col_name in enumerate(thermal_col_names):
                    idx = j + 1
                    if idx < len(cols):
                        try:
                            vals[col_name] = float(cols[idx])
                        except ValueError:
                            pass
                thermal_frames.append(ThermalFrame(ts_ms, vals))
            except (ValueError, IndexError):
                continue

except FileNotFoundError:
    print(f"  [WARN] thermal.log not found: {thermal_log}")


# ── CPU frequency log parsing ─────────────────────────────────────────────────
# cpufreq.log format:
# Line 0: timestamp_ms,cpu0_kHz,cpu1_kHz,...
# Lines 1+: data rows + ITER_START/END markers

class CpuFreqFrame:
    def __init__(self, ts_ms: float, freqs: list[float]):
        self.ts_ms = ts_ms
        self.freqs = freqs   # kHz per core

cpufreq_frames: list[CpuFreqFrame] = []
cpufreq_col_names: list[str] = []
cpufreq_iter_markers: list[dict] = []

try:
    with open(cpufreq_log) as f:
        header_line = f.readline().strip()
        all_cols = header_line.split(",")
        cpufreq_col_names = all_cols[1:]

        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("ITER_"):
                parts = raw.split(":")
                if len(parts) >= 5:
                    try:
                        cpufreq_iter_markers.append({
                            "event": parts[0], "model": parts[1],
                            "gen":   int(re.sub(r"\D", "", parts[2])),
                            "ctx":   int(re.sub(r"\D", "", parts[3])),
                            "ts_ms": float(parts[4]),
                        })
                    except ValueError:
                        pass
                continue
            cols = raw.split(",")
            if len(cols) < 2:
                continue
            try:
                ts_ms = float(cols[0])
                freqs = [float(cols[i+1]) for i in range(len(cpufreq_col_names)) if i+1 < len(cols)]
                cpufreq_frames.append(CpuFreqFrame(ts_ms, freqs))
            except (ValueError, IndexError):
                continue
except FileNotFoundError:
    pass


def cpufreq_stats_in_window(t0_ms: float, t1_ms: float) -> dict:
    """Returns avg and peak freq per core (MHz) for a time window."""
    frames = [f for f in cpufreq_frames if t0_ms <= f.ts_ms <= t1_ms]
    if not frames:
        return {}
    n_cores = len(cpufreq_col_names)
    stats = {}
    for i, col in enumerate(cpufreq_col_names):
        vals = [f.freqs[i] / 1000.0 for f in frames if i < len(f.freqs)]  # kHz → MHz
        if vals:
            stats[col] = {"avg_mhz": sum(vals)/len(vals), "peak_mhz": max(vals)}
    return stats


# ── Battery log parsing ───────────────────────────────────────────────────────
# battery.log has two kinds of rows:
#   CSV data: timestamp_ms,level_pct,temp_tenths_degC,voltage_mV,status
#   Snap markers: BATTERY_SNAP:<model>:<start|end>:<level>:<ts_ms>

class BatteryFrame:
    def __init__(self, ts_ms, level, temp_tenths, voltage, status):
        self.ts_ms       = ts_ms
        self.level       = level        # %
        self.temp_c      = temp_tenths / 10.0  # tenths → °C
        self.voltage_mv  = voltage
        self.status      = status

battery_frames: list[BatteryFrame] = []
battery_snaps:  dict[str, dict]    = {}  # model → {start_pct, end_pct}

try:
    with open(battery_log) as f:
        f.readline()   # skip header
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("BATTERY_SNAP:"):
                # BATTERY_SNAP:<model>:<start|end>:<level>:<ts_ms>
                parts = raw.split(":")
                if len(parts) >= 4:
                    model  = parts[1]
                    event  = parts[2]
                    try:
                        level = int(parts[3])
                        battery_snaps.setdefault(model, {})[event] = level
                    except ValueError:
                        pass
                continue
            cols = raw.split(",")
            if len(cols) < 5:
                continue
            try:
                battery_frames.append(BatteryFrame(
                    float(cols[0]), float(cols[1]),
                    float(cols[2]), float(cols[3]), cols[4]
                ))
            except (ValueError, IndexError):
                continue
except FileNotFoundError:
    pass


def frames_in_window(t0_ms: float, t1_ms: float) -> list[ThermalFrame]:
    return [f for f in thermal_frames if t0_ms <= f.ts_ms <= t1_ms]


def thermal_stats(frames: list[ThermalFrame]) -> dict:
    """Returns avg and peak per zone (in °C), plus a throttled flag."""
    if not frames:
        return {}

    zone_temps: dict[str, list[float]] = {}
    for fr in frames:
        for k, v in fr.vals.items():
            zone_temps.setdefault(k, []).append(v / 1000.0)  # millidegC → °C

    stats: dict = {}
    for k, temps in zone_temps.items():
        stats[k] = {
            "avg":  sum(temps) / len(temps),
            "peak": max(temps),
            "min":  min(temps),
        }

    # Throttling heuristic: skin or cpu zone peaks above 80°C
    all_peaks = [v["peak"] for v in stats.values()]
    stats["_throttled"] = any(p >= 80.0 for p in all_peaks)
    stats["_peak_any"]  = max(all_peaks) if all_peaks else 0.0
    return stats


# ── Timing log ────────────────────────────────────────────────────────────────
# MODEL_START:<name>:<epoch_s>
# MODEL_END:<name>:<epoch_s>
# Converted to ms for consistency with thermal log

model_windows: dict[str, dict] = {}

try:
    with open(timing_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MODEL_START:"):
                _, name, ts = line.split(":", 2)
                model_windows.setdefault(name, {})["start_ms"] = float(ts) * 1000
            elif line.startswith("MODEL_END:"):
                _, name, ts = line.split(":", 2)
                model_windows.setdefault(name, {})["end_ms"] = float(ts) * 1000
except FileNotFoundError:
    print(f"  [WARN] model_timing.log not found: {timing_log}")


# ── Per-iteration thermal windows (from ITER markers) ────────────────────────
# Build a map: (model, gen, ctx) → {start_ms, end_ms}
iter_windows: dict[tuple, dict] = {}
for m in iter_markers:
    key = (m["model"], m["gen"], m["ctx"])
    if m["event"] == "ITER_START":
        iter_windows.setdefault(key, {})["start_ms"] = m["ts_ms"]
    elif m["event"] == "ITER_END":
        iter_windows.setdefault(key, {})["end_ms"] = m["ts_ms"]


# ── Load aiperf results ───────────────────────────────────────────────────────
results: list[dict] = []

for json_path in sorted(glob.glob(f"{base_dir}/**/profile_export_aiperf.json", recursive=True)):
    rel   = os.path.relpath(json_path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) < 4:
        continue

    model_name = parts[0]
    gen_part   = parts[1]   # gen128
    ctx_part   = parts[2]   # ctx256

    try:
        gen = int(re.sub(r"\D", "", gen_part))
        ctx = int(re.sub(r"\D", "", ctx_part))
    except ValueError:
        continue

    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        continue

    # aiperf metric keys (may vary by version — try common names)
    def get_avg(d, *keys):
        for k in keys:
            if k in d and d[k]:
                v = d[k]
                if isinstance(v, dict):
                    return v.get("avg") or v.get("mean") or v.get("p50")
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    ttft   = get_avg(d, "time_to_first_token", "ttft")
    ttft_p50 = (d.get("time_to_first_token") or {}).get("p50")
    ttft_p90 = (d.get("time_to_first_token") or {}).get("p90")
    ttft_p99 = (d.get("time_to_first_token") or {}).get("p99")
    itl    = get_avg(d, "inter_token_latency", "itl")
    tps    = get_avg(d, "output_token_throughput_per_user", "output_token_throughput", "tps")
    req_lat = get_avg(d, "request_latency")

    # Thermal for this specific combo window
    iter_key = (model_name, gen, ctx)
    iter_win  = iter_windows.get(iter_key, {})
    t0 = iter_win.get("start_ms", 0)
    t1 = iter_win.get("end_ms", 9e18)
    combo_frames  = frames_in_window(t0, t1) if t0 > 0 else []
    combo_thermal = thermal_stats(combo_frames)

    

    # Extract useful zones — MTK Dimensity 7050 specific labels
    # Our discover_thermal_zones() uses keys: cpu, ap, charger, pa, mdpa, battery, soc
    def zone_peak(tstats, *patterns):
        for k, v in tstats.items():
            if k.startswith("_"):
                continue
            if any(p.lower() in k.lower() for p in patterns):
                return v["peak"]
        return None

    def zone_avg(tstats, *patterns):
        for k, v in tstats.items():
            if k.startswith("_"):
                continue
            if any(p.lower() in k.lower() for p in patterns):
                return v["avg"]
        return None

    peak_cpu     = zone_peak(combo_thermal, "cpu")
    peak_ap      = zone_peak(combo_thermal, "ap")        # AP die temp
    peak_charger = zone_peak(combo_thermal, "charger")   # PMIC/charger hotspot
    peak_battery = zone_peak(combo_thermal, "battery")   # battery NTC
    peak_any     = combo_thermal.get("_peak_any", 0)
    throttled    = combo_thermal.get("_throttled", False)

    results.append({
        "model":        model_name,
        "gen":          gen,
        "ctx":          ctx,
        "ttft_avg":     ttft,
        "ttft_p50":     ttft_p50,
        "ttft_p90":     ttft_p90,
        "ttft_p99":     ttft_p99,
        "itl_avg":      itl,
        "tps":          tps,
        "req_lat":      req_lat,
        "peak_cpu_c":   peak_cpu,
        "peak_ap_c":    peak_ap,
        "peak_chg_c":   peak_charger,
        "peak_batt_c":  peak_battery,
        "peak_any_c":   peak_any if peak_any else None,
        "throttled":    throttled,
    })

results.sort(key=lambda r: (r["model"], r["gen"], r["ctx"]))


# ── Per-model thermal summary (full model window) ─────────────────────────────
model_thermal_summary: dict[str, dict] = {}

for model_name, win in model_windows.items():
    t0 = win.get("start_ms", 0)
    t1 = win.get("end_ms", 9e18)
    frames = frames_in_window(t0, t1)
    tstats = thermal_stats(frames)

    model_thermal_summary[model_name] = {
        "peak_cpu":     zone_peak(tstats, "cpu"),
        "peak_ap":      zone_peak(tstats, "ap"),
        "peak_charger": zone_peak(tstats, "charger"),
        "peak_battery": zone_peak(tstats, "battery"),
        "peak_any":     tstats.get("_peak_any", 0),
        "throttled":    tstats.get("_throttled", False),
        "n_frames":     len(frames),
    }


# ── Per-model best TTFT ───────────────────────────────────────────────────────
best_ttft: dict[str, dict] = {}
for r in results:
    m = r["model"]
    if r["ttft_avg"] is not None:
        if m not in best_ttft or r["ttft_avg"] < best_ttft[m]["ttft_avg"]:
            best_ttft[m] = r


# ── Write report ──────────────────────────────────────────────────────────────
def fmt(v, fmt_str, fallback="—"):
    return format(v, fmt_str) if v is not None else fallback


lines: list[str] = []
def L(s=""): lines.append(s)

L(f"# Android LLM Benchmark — {device_name}")
L()
L(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
L(f"**Device:** {device_name}  ")
L(f"**Framework:** {framework}  **Backend:** llama-server (arm64, CPU)  **Context:** {ctx_size} tokens  **Concurrency:** 1  ")
L(f"**Sweep:** prompt in {{{','.join(str(p) for p in sorted(set(r['ctx'] for r in results)))}}}  "
  f"gen in {{{','.join(str(g) for g in sorted(set(r['gen'] for r in results)))}}}  ")
L(f"**Thermal:** polled at 500ms via `adb shell /sys/class/thermal/`")
L()

L("> **Note on power measurement:** Android does not expose reliable per-component power to third-party tools.")
L("> `Battery Manager API` is unreliable under GPU load (confirmed by arXiv:2603.23640).")
L("> Thermal data is from `/sys/class/thermal/` zone readings (°C). Power metrics are not reported.")
L()

if skipped:
    L("## Skipped Models")
    L()
    for s in skipped:
        L(f"- {s}")
    L()

# Full results table
L("## Full Results")
L()
L("| Model | Prompt (tok) | Gen (tok) | TTFT avg (ms) | TTFT p50 (ms) | TTFT p90 (ms) | ITL avg (ms) | Tok/s | CPU (°C) | AP die (°C) | Charger (°C) | Battery (°C) | Throttled |")
L("|-------|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")

for r in results:
    L(
        f"| {r['model']} "
        f"| {r['ctx']} "
        f"| {r['gen']} "
        f"| {fmt(r['ttft_avg'],    '.0f')} "
        f"| {fmt(r['ttft_p50'],    '.0f')} "
        f"| {fmt(r['ttft_p90'],    '.0f')} "
        f"| {fmt(r['itl_avg'],     '.2f')} "
        f"| {fmt(r['tps'],         '.2f')} "
        f"| {fmt(r['peak_cpu_c'],  '.1f')} "
        f"| {fmt(r['peak_ap_c'],   '.1f')} "
        f"| {fmt(r['peak_chg_c'],  '.1f')} "
        f"| {fmt(r['peak_batt_c'], '.1f')} "
        f"| {'⚠️ YES' if r['throttled'] else 'No'} |"
    )

L()

# Per-model best TTFT
L("## Per-Model Best TTFT (lowest avg, across all prompt×gen combos)")
L()
L("| Model | Best TTFT avg (ms) | TTFT p50 (ms) | TTFT p90 (ms) | Prompt (tok) | Gen (tok) | Tok/s |")
L("|-------|---:|---:|---:|:---:|:---:|---:|")

for model_name in sorted(best_ttft.keys()):
    b = best_ttft[model_name]
    L(
        f"| {b['model']} "
        f"| **{fmt(b['ttft_avg'],'.0f')}** "
        f"| {fmt(b['ttft_p50'],  '.0f')} "
        f"| {fmt(b['ttft_p90'],  '.0f')} "
        f"| {b['ctx']} "
        f"| {b['gen']} "
        f"| {fmt(b['tps'],       '.2f')} |"
    )

L()

# Thermal summary
L("## Thermal Summary (per-model, full benchmark window)")
L()
L("> Throttling heuristic: any zone peaked ≥ 80°C during the model's run.")
L("> MTK zones: `mtktscpu` = CPU cluster, `mtktsAP` = AP die, `mtktscharger` = PMIC, `mtktsbattery` = battery NTC.")
L()
L("| Model | Peak CPU (°C) | Peak AP die (°C) | Peak Charger (°C) | Peak Battery (°C) | Peak Any (°C) | Throttled | Frames |")
L("|-------|---:|---:|---:|---:|---:|:---:|---:|")

for model_name in sorted(model_thermal_summary.keys()):
    t = model_thermal_summary[model_name]
    L(
        f"| {model_name} "
        f"| {fmt(t['peak_cpu'],     '.1f')} "
        f"| {fmt(t['peak_ap'],      '.1f')} "
        f"| {fmt(t['peak_charger'], '.1f')} "
        f"| {fmt(t['peak_battery'], '.1f')} "
        f"| {fmt(t['peak_any'],     '.1f')} "
        f"| {'⚠️ YES' if t['throttled'] else 'No'} "
        f"| {t['n_frames']} |"
    )

L()

# CPU frequency summary per model
if cpufreq_frames:
    L("## CPU Frequency Summary (per-model window, avg across cores)")
    L()
    L("> **Dimensity 7050 layout:** cpu0–cpu1 = Cortex-A78 (big, 2.6 GHz max) · cpu2–cpu7 = Cortex-A55 (little, 2.0 GHz max)")
    L("> Per arXiv:2410.03613: optimal llama.cpp thread count = 2 (big cores only). Watch cpu0/cpu1 — if they're staying below 2600 MHz under load, DVFS is throttling.")
    L()

    # Figure out which cores are big vs little from peak freqs
    # (big cores run at higher max freq — typically > 2000 MHz for A78)
    col_header = " | ".join(f"{c}" for c in cpufreq_col_names)
    L(f"| Model | {col_header} | Peak any (MHz) |")
    L(f"|-------|{'|'.join(['---:' for _ in cpufreq_col_names])}|---:|")

    for model_name, win in model_windows.items():
        t0 = win.get("start_ms", 0)
        t1 = win.get("end_ms", 9e18)
        cstats = cpufreq_stats_in_window(t0, t1)
        if not cstats:
            continue
        avg_cols = " | ".join(
            f"{cstats[c]['avg_mhz']:.0f}" if c in cstats else "—"
            for c in cpufreq_col_names
        )
        peak_any = max((v["peak_mhz"] for v in cstats.values()), default=0)
        L(f"| {model_name} | {avg_cols} | {peak_any:.0f} |")

    L()

# Battery drain table
if battery_snaps:
    L("## Battery Drain (per model)")
    L()
    L("> `dumpsys battery` drain — rough proxy only, not real power draw.")
    L()
    L("| Model | Start (%) | End (%) | Drain (%) |")
    L("|-------|---:|---:|---:|")
    total_drain = 0
    for model_name in sorted(battery_snaps.keys()):
        s = battery_snaps[model_name]
        start = s.get("start")
        end   = s.get("end")
        drain = (start - end) if (start is not None and end is not None) else None
        if drain is not None:
            total_drain += drain
        L(
            f"| {model_name} "
            f"| {fmt(start, '.0f')} "
            f"| {fmt(end,   '.0f')} "
            f"| {fmt(drain, '.0f')} |"
        )
    if total_drain > 0:
        L(f"| **Total** | | | **{total_drain:.0f}** |")
    L()

# Thermal zones discovered
zones_file = os.path.join(base_dir, "thermal_zones.txt")
if os.path.exists(zones_file):
    L("## Thermal Zones Used")
    L()
    L("```")
    with open(zones_file) as f:
        for line in f:
            L(line.rstrip())
    L("```")
    L()

# Device info block
info_file = os.path.join(base_dir, "device_info.txt")
if os.path.exists(info_file):
    L("## Device Info")
    L()
    L("```")
    with open(info_file) as f:
        for line in f:
            L(line.rstrip())
    L("```")
    L()

L("---")
L(f"*Generated by `generate_report.py` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
L(f"*{len(results)} result rows  |  {len(model_thermal_summary)} models  |  {len(skipped)} skipped*")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  Report → {report_path}")
print(f"  {len(results)} rows  |  {len(model_thermal_summary)} models  |  {len(skipped)} skipped")
if model_thermal_summary:
    throttled_count = sum(1 for t in model_thermal_summary.values() if t["throttled"])
    print(f"  Thermal throttling detected in {throttled_count}/{len(model_thermal_summary)} models")