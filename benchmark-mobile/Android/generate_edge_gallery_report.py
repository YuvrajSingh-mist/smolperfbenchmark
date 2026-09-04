#!/usr/bin/env python3
"""MoE-blog-style summary report for a benchmark-edge-gallery-chat.sh run.

Usage: generate_edge_gallery_report.py <run_base> [<out_md>]

Walks <run_base>/gen<G>/ctx<P>/profile_export_aiperf.json for every combo, reads
the run's thermal.log (labeled SoC CSV), and writes one REPORT.md with a
latency/throughput table per combo + a real-SoC thermal summary + an honest
power/tok-J note (unrooted tablet → no W rail).

thermal.log columns (from benchmark-edge-gallery-chat.sh):
  ts_epoch_s,cpu_max_c,gpu_c,npu_c,ddr_c,pmic_c,batt_c,cpu_max_mhz,rss_mb
"""
import csv, json, os, sys, glob

BASE = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "REPORT.md")

def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def stat(D, name, field):
    m = D.get(name)
    return m.get(field) if isinstance(m, dict) else None

def num(x, nd=1):
    if x is None:
        return "—"
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "—"

combos = []
for jp in sorted(glob.glob(os.path.join(BASE, "gen*", "ctx*", "profile_export_aiperf.json"))):
    parts = jp.split(os.sep)
    gen = next((p for p in parts if p.startswith("gen")), "?")
    ctx = next((p for p in parts if p.startswith("ctx")), "?")
    d = load_json(jp)
    if not d:
        continue
    combos.append((ctx, gen, d, jp))

# ---- thermal summary ----
rows = []
th = os.path.join(BASE, "thermal.log")
if os.path.exists(th):
    with open(th) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split(",")
            if len(p) >= 8:
                try:
                    rows.append([float(x) for x in p[:8]])
                except ValueError:
                    pass

def mx(i):
    vals = [r[i] for r in rows if i < len(r)]
    return max(vals) if vals else None

def throttle_verdict():
    if len(rows) < 3:
        return "—"
    freqs = [r[7] for r in rows]
    peak = max(freqs)
    tail = freqs[len(freqs)//2:]
    tail_avg = sum(tail)/len(tail)
    if peak and tail_avg < 0.6*peak and (mx(1) or 0) > 70:
        return f"Yes — freq collapsed to ~{tail_avg:.0f} MHz of {peak:.0f} MHz peak"
    return f"No — sustained ~{tail_avg:.0f} MHz (peak {peak:.0f} MHz)"

# ---- battery-slope (whole-device) ----
batt = {}
bl = os.path.join(BASE, "battery.log")
if os.path.exists(bl):
    for line in open(bl):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            batt[k.strip()] = v.strip()
avg_w = batt.get("avg_whole_device_w", "")
b0 = batt.get("battery_level_start", "")
b1 = batt.get("battery_level_end", "")

# ---- whole-run output tokens + time across combos ----
tot_out = sum(float((stat(d, "total_output_tokens", "avg") or 0)) for _, _, d, _ in combos)
sum_dur = sum(float((stat(d, "benchmark_duration", "avg") or 0)) for _, _, d, _ in combos)
sys_tokj = ""
try:
    if avg_w not in ("", "na") and float(avg_w) > 0 and sum_dur > 0 and tot_out > 0:
        sys_tokj = f"{tot_out / (float(avg_w) * sum_dur):.3f}"
except Exception:
    sys_tokj = ""

L = []
A = L.append
A("# Edge Gallery (LiteRT-LM) — Multi-Combo Benchmark Summary")
A("")
A("> Metric layout mirrors the Mac Mini M4 MoE blog: latency/throughput from `aiperf profile` (client-side token counting) + a **real-SoC thermal summary** (device tsens) + a coarse **whole-device battery-slope W**. True decode-only W / tok/J needs a power rail, which unrooted Android does not expose.")
A("")
A("**Run dir:** `" + BASE + "`")
A("")
if combos:
    A("## 1. Per-combo latency & throughput")
    A("")
    A("| Combo | TTFT p50/p90/p99 (ms) | ITL p50 (ms) | Decode (tok/s) | Prefill (tok/s) | E2E out (tok/s) | In/Out seq | Err% |")
    A("|---|---:|---:|---:|---:|---:|---|---:|")
    for ctx, gen, d, jp in combos:
        ttft = f"{num(stat(d,'time_to_first_token','p50'),0)}/{num(stat(d,'time_to_first_token','p90'),0)}/{num(stat(d,'time_to_first_token','p99'),0)}"
        itl = num(stat(d, 'inter_token_latency', 'p50'), 0)
        dec = num(stat(d, 'active_decode_throughput', 'avg'))
        pre = num(stat(d, 'active_prefill_throughput', 'avg'))
        e2e = num(stat(d, 'e2e_output_token_throughput', 'avg'))
        i_n = num(stat(d, 'input_sequence_length', 'avg'), 0)
        o_n = num(stat(d, 'output_sequence_length', 'avg'), 0)
        err = num(stat(d, 'request_error_rate', 'avg'))
        label = f"{ctx[3:]}+{gen[3:]}"
        A(f"| {label} | {ttft} | {itl} | {dec} | {pre} | {e2e} | {i_n}/{o_n} | {err} |")
    A("")
    A("> ⚠️ If `--concurrency > 1` was used, TTFT/prefill include queue wait: Edge Gallery keeps ONE LiteRT conversation per model and the server serializes requests. Keep `--concurrency 1` for clean per-request latency; ITL and decode tok/s are unaffected either way.")
    A("")
else:
    A("*(no per-combo aiperf JSONs found under gen*/ctx/)*")
    A("")

A("## 2. Power, tok/J & energy")
A("")
A("| Metric | value |")
A("|---|---:|")
if avg_w not in ("", "na"):
    A(f"| Whole-device avg W (battery-slope {b0}%→{b1}%) | **{avg_w} W** (SYSTEM-wide — includes screen/OS; set BATTERY_WH for the device) |")
    if sys_tokj:
        A(f"| Whole-device tok/J (system) | **{sys_tokj}** ≈ total output tokens ÷ (avg W × run s) |")
    A("| Decode-only W / tok/J (Mac-blog style) | **not measurable** — needs a power rail unrooted Android cannot expose |")
else:
    A("| Avg power (W) | **battery-slope not available** — set `BATTERY_WH` (device Wh) and run long enough for a ≥1% battery drop; unplug charger |")
    A("| Decode-only W / tok/J | **not measurable** — no power rail on unrooted Android (no `powermetrics`) |")
A("")
A("## 3. Thermal summary (real SoC temps, whole run)")
A("")
A("| Block (device zone) | Peak °C |")
A("|---|---:|")
A(f"| CPU hottest (`*cpu*`/fallback) | {num(mx(1))} |")
A(f"| GPU (`*gpu*`) | {num(mx(2))} |")
A(f"| NPU (`*npu*`/dsp) | {num(mx(3))} |")
A(f"| DDR (`*ddr*`) | {num(mx(4))} |")
A(f"| PMIC (`*pmic*`) | {num(mx(5))} |")
A(f"| Battery (`*batt*`) | {num(mx(6))} |")
A(f"| CPU max freq | {num(mx(7),0)} MHz |")
A(f"| **Throttled?** | **{throttle_verdict()}** |")
A(f"| Thermal samples | {len(rows)} |")
A("")
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, f"({len(combos)} combos, {len(rows)} thermal samples, avg_w={avg_w})")
