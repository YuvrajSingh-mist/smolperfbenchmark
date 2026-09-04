#!/bin/bash
# benchmark-edge-gallery-chat.sh
#
# Benchmarks the Edge Gallery (LiteRT-LM) fork's OpenAI-compatible chat server
# on an Android tablet (adb-connected). The server runs IN-PROCESS inside the
# app (openserver/OpenAiLlmServer.kt), serving /v1/chat/completions over SSE —
# so no llama-server binary is needed, same as benchmark-pipette-chat.sh.
#
#   app (Xiaomi Pad 6, com.google.aiedge.gallery)
#     └─ OpenAI-compatible server @ device:808x (started via app Settings →
#          "Local benchmark server" toggle)
#          │  adb forward tcp:<host> → tcp:<device>
#          ▼
#        host aiperf 0.13 --endpoint-type chat --url http://localhost:<host>
#
# The model is downloaded in the app (AI Chat → model → Download) and the
# server is started from **Settings → Local benchmark server**. This script
# auto-detects which port the server bound (it scans 8080..8089 because the
# app falls back when 8080 is taken, e.g. by pipette), then runs the full
# aiperf sweep and collects a labeled thermal/cpufreq/RSS log (real SoC °C)
# the whole run — the same chart/report artifacts as the other mobile benches.
#
# Prep (one-time):
#   1. Install the fork build, launch it, download a PUBLIC model in AI Chat
#      (e.g. Qwen2.5-1.5B-Instruct; Gemma3-1B-IT is HF-gated and OAuth is
#      broken in debug builds → it won't download).
#   2. Settings → toggle "Local benchmark server" ON (note the port shown).
#   3. (optional) adb forward tcp:<port> tcp:<port> — the script does this.
#
# Usage:
#   bash benchmark-edge-gallery-chat.sh
#   bash benchmark-edge-gallery-chat.sh --model Qwen2.5-1.5B-Instruct --reqs 10
#   bash benchmark-edge-gallery-chat.sh --prompt-lengths "128 512 2048" --gen-lengths "64 128" --concurrency 1
#   bash benchmark-edge-gallery-chat.sh --dry-run
#   bash benchmark-edge-gallery-chat.sh --skip-smoke
#
# Requirements:
#   - adb (device authorised), fork app installed + server toggled ON in Settings
#   - aiperf 0.13 (this repo's venv: ~/Desktop/smolbenchmark/venv/bin/aiperf)

set -euo pipefail

# ── Auto-relaunch inside tmux (survives terminal close, like the other benches) ──
if [ -z "${TMUX:-}" ]; then
    SESSION="edge-gallery-bench"
    SELF="$(realpath "$0")"
    ARGS="$(printf '%q ' "$@")"
    ENV_FWDS=""
    [ -n "${AIPERF_BIN:-}" ] && ENV_FWDS+="export AIPERF_BIN=$(printf '%q' "$AIPERF_BIN"); "
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "bash -c '${ENV_FWDS}bash ${SELF} ${ARGS}; echo; echo === Done — press Enter to exit ===; read'"
    echo "Launched in tmux session '$SESSION'."
    echo "Attach with:  tmux attach -t $SESSION"
    exit 0
fi

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_LABEL="Qwen2.5-1.5B-Instruct"
BACKEND=""                 # "cpu" | "gpu" | "" (app/server default). Sent as a per-request
                           # `accelerator` override; the server reloads the engine when it changes.
TOKENIZER=""              # aiperf tokenizer (must match the model). Auto-map if unset.
BATTERY_WH=""             # device battery capacity in Wh for coarse whole-device avg-W (e.g. 34.2)
REQS=10
CONCURRENCY=1            # Edge Gallery serializes on ONE LiteRT conversation —
                         # keep 1 for clean per-request latency; >1 measures queueing.
ONLY_COMBO=""
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
REQUEST_TIMEOUT=300
COOLDOWN_COMBO=10        # s between combos (thermal bleed)
SERVER_STARTUP_TIMEOUT=20
THERMAL_INTERVAL="${THERMAL_INTERVAL:-1}"  # s between SoC samples; 0.5 = Jetson tegrastats parity

PROMPT_LENGTHS=(128 512 1024)
GEN_LENGTHS=(64 128)
DETECT_PORTS="8080 8081 8082 8083 8084 8085 8086 8087 8088 8089"

ADB="adb"
SERIAL=""
AIPERF_BIN="${AIPERF_BIN:-$(which aiperf 2>/dev/null || echo "$HOME/Desktop/smolbenchmark/venv/bin/aiperf")}"

BASE_ARTIFACT=""
THERMAL_LOG=""
DEVICE_INFO_FILE=""

# ── Args ──────────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --model)          MODEL_LABEL="$2"; shift 2 ;;
        --backend)        case "$2" in cpu|gpu) BACKEND="$2" ;; *) echo "--backend must be cpu or gpu (got '$2')"; exit 1 ;; esac; shift 2 ;;
        --tokenizer)      TOKENIZER="$2"; shift 2 ;;
        --reqs)           REQS="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY="$2"; shift 2 ;;
        --prompt-lengths) PROMPT_LENGTHS=($2); shift 2 ;;
        --gen-lengths)    GEN_LENGTHS=($2); shift 2 ;;
        --only-combo)     ONLY_COMBO="$2"; shift 2 ;;
        --serial)         SERIAL="$2"; shift 2 ;;
        --skip-smoke)     SKIP_SMOKE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --resume)         RESUME_DIR="$2"; shift 2 ;;
        -h|--help)        sed -n '1,45p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -n "$SERIAL" ]; then ADB="adb -s $SERIAL"; fi

# Map Edge Gallery model name -> HF tokenizer repo (override with --tokenizer).
tokenizer_for() {
    [ -n "$TOKENIZER" ] && { echo "$TOKENIZER"; return; }
    case "$MODEL_LABEL" in
        Qwen*)           echo "Qwen/Qwen2.5-1.5B-Instruct" ;;
        DeepSeek*)       echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ;;
        Gemma3-1B-IT)    echo "google/gemma-3-1b-it" ;;
        *)               echo "" ;;
    esac
}
RESOLVED_TOKENIZER=$(tokenizer_for)
if [ -z "$RESOLVED_TOKENIZER" ]; then
    echo "WARN: no auto tokenizer for '$MODEL_LABEL' — pass --tokenizer <hf-repo>"
fi

# ── Resolve artifact dir ──────────────────────────────────────────────────────
if [ -n "$RESUME_DIR" ]; then
    BASE_ARTIFACT="$RESUME_DIR"
    echo "  [RESUME] Reusing artifact dir: $BASE_ARTIFACT"
else
    BASE_ARTIFACT="${SCRIPT_DIR}/artifacts/edge-gallery-chat/$(date +%Y%m%d-%H%M)"
fi
THERMAL_LOG="${BASE_ARTIFACT}/thermal.log"
DEVICE_INFO_FILE="${BASE_ARTIFACT}/device-info.txt"

log()  { printf '%s\n' "$*"; }
err()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
ts()   { date '+%Y-%m-%d %H:%M:%S'; }

# ── Prereqs ───────────────────────────────────────────────────────────────────
command -v "${ADB%% *}" >/dev/null || err "adb not found in PATH"
[ -x "$AIPERF_BIN" ] || err "aiperf not found at $AIPERF_BIN"

DEVICE=$($ADB devices | awk 'NR==2{print $1}')
[ -z "$DEVICE" ] && err "No adb device attached"

echo "=== Edge Gallery (LiteRT-LM) chat benchmark ==="
echo "Device:            $DEVICE"
echo "Model:             $MODEL_LABEL  (must be downloaded in the app)"
echo "Backend:            ${BACKEND:-server-default (allowlist default, usually gpu)}"
echo "Requests/combo:    $REQS   concurrency: $CONCURRENCY"
echo "Prompt lengths:    ${PROMPT_LENGTHS[*]}"
echo "Gen lengths:       ${GEN_LENGTHS[*]}"
echo "Artifact dir:      $BASE_ARTIFACT"
mkdir -p "$BASE_ARTIFACT"
$ADB shell getprop ro.product.model > "$DEVICE_INFO_FILE" 2>/dev/null || true
$ADB shell getprop ro.hardware.egl  >> "$DEVICE_INFO_FILE" 2>/dev/null || true

# ── Find the in-app server port (it scans 8080.. since 8080 is often pipette) ──
find_server_port() {
    for p in $DETECT_PORTS; do
        $ADB forward "tcp:${p}" "tcp:${p}" >/dev/null 2>&1 || true
        # Match by model name so we don't mistake pipette's OpenAI server for Edge Gallery.
        body=$(curl -s "http://localhost:${p}/v1/models" --max-time 3 2>/dev/null || true)
        if echo "$body" | grep -qF "$MODEL_LABEL"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

HOST_PORT=""
if [ "$DRY_RUN" = "0" ]; then
    for i in $(seq 1 "$SERVER_STARTUP_TIMEOUT"); do
        HOST_PORT=$(find_server_port) && break
        log "  Waiting for the Edge Gallery server (Settings → Local benchmark server)… ($i)"
        sleep 1
    done
    [ -n "$HOST_PORT" ] || err "Server not found on ports ${DETECT_PORTS}. Open app → Settings → toggle 'Local benchmark server' ON, and ensure '$MODEL_LABEL' is downloaded."
    log "  Server found on port ${HOST_PORT} (adb forward tcp:${HOST_PORT} → device)"
else
    HOST_PORT=8081
fi
SERVER_URL="http://localhost:${HOST_PORT}"

# ── On-device labeled SoC thermal / cpufreq / RSS sampler (whole run) ─────────
# Reads Qualcomm tsens zone TYPES ON THE DEVICE (unrooted; only the battery
# *power* rail is Permission-denied — not needed) and streams CSV lines back
# over adb into thermal.log. Running the poll loop on-device (pushed script)
# removes the host-adb round-trip, so sampling can reach Jetson-grade density
# (tegrastats --interval 500 ⇒ THERMAL_INTERVAL=0.5) instead of a slow host-side
# cadence. Set THERMAL_INTERVAL to your target seconds.
# Columns: ts_epoch_s,cpu_max_c,gpu_c,npu_c,ddr_c,pmic_c,batt_c,cpu_max_mhz,rss_mb
APP_PKG="${APP_PKG:-com.google.aiedge.gallery}"
SAMPLER_PIDFILE="/tmp/edge_gallery_thermal.pid"
DEVICE_POLL="/data/local/tmp/soc_thermal_poll.sh"
DEVICE_OUT="/data/local/tmp/soc_thermal.out"

start_sampler() {
    [ "$DRY_RUN" = "1" ] && return 0
    local tmp_poll
    tmp_poll=$(mktemp /tmp/soc_thermal_poll_XXXX.sh)
    cat > "$tmp_poll" <<'DEVPOLL'
#!/system/bin/sh
iv=$1; pkg=$2; out=$3
printf '%s\n' "# ts_epoch,cpu_max_c,gpu_c,npu_c,ddr_c,pmic_c,batt_c,cpu_max_mhz,rss_mb" > "$out"
while true; do
  pmax=0; gmax=0; npu=0; ddr=0; pmic=0; batt=0; fmax=0; best=0; allmax=0
  for z in /sys/class/thermal/thermal_zone[0-9]*; do
    IFS= read -r t < "$z/type" 2>/dev/null || continue
    IFS= read -r v < "$z/temp" 2>/dev/null || continue
    [ -n "$t" ] || continue
    [ -n "$v" ] || continue
    case "$v" in ''|*[!0-9-]*) continue ;; esac
    [ "$v" -gt "$allmax" ] && allmax=$v
    # Device-agnostic keyword classification (Qualcomm / MediaTek / Exynos / ...).
    case "$t" in
      *gpu*)                     [ "$v" -gt "$gmax" ] && gmax=$v ;;
      *npu*|*hexagon*|*-dsp*|*apus*) npu=$v ;;
      *cpu*|*cpuss*|*cluster*)   [ "$v" -gt "$pmax" ] && pmax=$v ;;
      *ddr*)                     ddr=$v ;;
      *pmic*|pm8*_tz)            pmic=$v ;;
      *batt*)                    batt=$v ;;
    esac
  done
  # Non-Qualcomm SoCs often omit cpu-*/gpuss-* zone types — fall back to the
  # hottest sensor overall so a run still reports a temperature.
  [ "$pmax" -eq 0 ] && [ "$allmax" -gt 0 ] && pmax=$allmax
  [ "$gmax" -eq 0 ] && [ "$allmax" -gt 0 ] && gmax=$allmax
  for f in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_cur_freq; do
    IFS= read -r v < "$f" 2>/dev/null || continue
    [ -n "$v" ] && [ "$v" -gt "$fmax" ] && fmax=$v
  done
  for p in $(pgrep -f "$pkg" 2>/dev/null); do
    r=$(awk '/VmRSS/{print $2}' "/proc/$p/status" 2>/dev/null)
    [ "${r:-0}" -gt "$best" ] && best=$r
  done
  ts=$(date +%s)
  printf '%s\n' "$ts,$((pmax/1000)),$((gmax/1000)),$((npu/1000)),$((ddr/1000)),$((pmic/1000)),$((batt/1000)),$((fmax/1000)),$((best/1024))" >> "$out"
  sleep "$iv"
done
DEVPOLL
    $ADB push "$tmp_poll" "$DEVICE_POLL" >/dev/null 2>&1
    $ADB shell chmod +x "$DEVICE_POLL"
    rm -f "$tmp_poll"
    $ADB shell "rm -f $DEVICE_OUT"
    # Start detached on-device; it appends to DEVICE_OUT at full rate.
    $ADB shell "nohup $DEVICE_POLL $THERMAL_INTERVAL '$APP_PKG' $DEVICE_OUT >/dev/null 2>&1 &"
    # Host mirror loop: copy the on-device CSV into THERMAL_LOG every 2s.
    {
        while true; do
            $ADB shell "cat $DEVICE_OUT 2>/dev/null" | tr -d '\r' > "$THERMAL_LOG"
            sleep 2
        done
    } >/dev/null 2>&1 &
    echo $! > "$SAMPLER_PIDFILE"
    log "  On-device SoC poller ${THERMAL_INTERVAL}s → ${DEVICE_OUT} → $THERMAL_LOG"
}

stop_sampler() {
    if [ -f "$SAMPLER_PIDFILE" ]; then
        kill "$(cat "$SAMPLER_PIDFILE")" 2>/dev/null || true
        rm -f "$SAMPLER_PIDFILE"
    fi
    $ADB shell "pkill -f soc_thermal_poll.sh" 2>/dev/null || true
    # Final flush of whatever the device file holds.
    $ADB shell "cat $DEVICE_OUT 2>/dev/null" | tr -d '\r' > "$THERMAL_LOG" 2>/dev/null || true
    $ADB shell "rm -f $DEVICE_OUT" 2>/dev/null || true
}
trap stop_sampler EXIT

# ── Per-combo sweep ───────────────────────────────────────────────────────────
run_combo() {
    local ct="${1}"; local gen="${2}"
    local combo_dir="${BASE_ARTIFACT}/gen${gen}/ctx${ct}"
    mkdir -p "$combo_dir"

    if [ "$DRY_RUN" = "1" ]; then
        log "  [DRY] aiperf --url ${SERVER_URL} --model ${MODEL_LABEL} ctx=${ct} gen=${gen} reqs=${REQS} cc=${CONCURRENCY}"
        return 0
    fi

    # Smoke: one real streaming completion before the sweep (carries --backend so
    # the very first engine load uses the requested accelerator).
    if [ "$SKIP_SMOKE" = "0" ]; then
        smoke_body="{\"model\":\"${MODEL_LABEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence\"}],\"stream\":true,\"max_tokens\":16"
        [ -n "$BACKEND" ] && smoke_body="${smoke_body},\"accelerator\":\"${BACKEND}\""
        smoke_body="${smoke_body}}"
        code=$(curl -sN "${SERVER_URL}/v1/chat/completions" --max-time "$REQUEST_TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "$smoke_body" \
            -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        log "  [SMOKE] HTTP ${code} (200 expected) — ${MODEL_LABEL} ctx=${ct} gen=${gen}${BACKEND:+ backend=$BACKEND}"
    fi

    log "  aiperf sweep ctx=${ct} gen=${gen} reqs=${REQS} cc=${CONCURRENCY}${BACKEND:+ backend=$BACKEND}"
    export AIPERF_SERVICE_REGISTRATION_TIMEOUT=300
    local -a extra_inputs=()
    [ -n "$BACKEND" ] && extra_inputs+=(--extra-inputs "accelerator:$BACKEND")
    local -a tok=()
    [ -n "$RESOLVED_TOKENIZER" ] && tok=(--tokenizer "$RESOLVED_TOKENIZER")
    "$AIPERF_BIN" profile \
        --endpoint-type chat \
        --url "${SERVER_URL}" \
        --model-names "${MODEL_LABEL}" \
        --streaming \
        "${tok[@]}" \
        --prompt-input-tokens-mean "$ct" \
        --prompt-output-tokens-mean "$gen" \
        --request-count "$REQS" \
        --concurrency "$CONCURRENCY" \
        "${extra_inputs[@]}" \
        --artifact-dir "$combo_dir" \
        || log "  aiperf failed (ctx=${ct} gen=${gen})"
    sleep "$COOLDOWN_COMBO"
}

# ── Run ───────────────────────────────────────────────────────────────────────
BATTERY_LOG="${BASE_ARTIFACT}/battery.log"
battery_level() { $ADB shell "dumpsys battery 2>/dev/null | grep '^  level:'" 2>/dev/null | awk '{print $2}' | tr -d '\r'; }

BAT0=""; BAT1=""; T0=$(date +%s); T1="$T0"
if [ "$DRY_RUN" = "0" ]; then
    BAT0=$(battery_level)
    start_sampler
fi

for ct in "${PROMPT_LENGTHS[@]}"; do
    for gen in "${GEN_LENGTHS[@]}"; do
        if [ -n "$ONLY_COMBO" ] && [ "$ONLY_COMBO" != "${ct}:${gen}" ]; then continue; fi
        run_combo "$ct" "$gen"
    done
done

if [ "$DRY_RUN" = "0" ]; then
    stop_sampler
    T1=$(date +%s); BAT1=$(battery_level)
fi
trap - EXIT

# Coarse whole-device avg W from battery drain — honest: SYSTEM-WIDE (includes
# screen/OS), not decode-only like Jetson/M4. Needs BATTERY_WH set AND a >=1%
# drop during the run; otherwise reported as na.
AVG_W="na"
if [ "$DRY_RUN" = "0" ] && [ -n "$BATTERY_WH" ] && [ -n "$BAT0" ] && [ -n "$BAT1" ] && [ "$BAT1" -lt "$BAT0" ]; then
    DUR=$((T1 - T0))
    AVG_W=$(awk -v d="$((BAT0 - BAT1))" -v wh="$BATTERY_WH" -v s="$DUR" 'BEGIN{ if (s>0 && d>0) printf "%.2f", (d/100.0)*wh/(s/3600.0); else print "na" }')
fi
{ echo "battery_level_start=$BAT0"; echo "battery_level_end=$BAT1"; echo "duration_s=$((T1 - T0))"; echo "battery_wh=$BATTERY_WH"; echo "avg_whole_device_w=$AVG_W"; } > "$BATTERY_LOG"

echo
echo "=== Done. Artifacts in ${BASE_ARTIFACT} ==="
echo "    thermal.log:  $(tail -1 "$THERMAL_LOG" 2>/dev/null)"
echo "    battery:      ${BAT0}% -> ${BAT1}% over $((T1 - T0))s → avg whole-device W: ${AVG_W} (na unless BATTERY_WH set + ≥1% drop)"
# Optional MoE-blog-style summary report.
if [ -f "${SCRIPT_DIR}/generate_edge_gallery_report.py" ]; then
    python3 "${SCRIPT_DIR}/generate_edge_gallery_report.py" "$BASE_ARTIFACT" "$BASE_ARTIFACT/REPORT.md" 2>/dev/null \
        && echo "    REPORT.md written (MoE-blog-style summary)"
fi
