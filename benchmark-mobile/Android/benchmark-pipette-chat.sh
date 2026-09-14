#!/opt/homebrew/bin/bash
# benchmark-pipette-chat.sh
#
# Benchmarks the Pipette app's native OpenAI-compatible chat server on an
# Android tablet (adb-connected) — no llama-server binary needed, because the
# server runs IN-PROCESS inside the app (crates/pipette-android/src/server.rs),
# serving the same /v1/chat/completions surface aiperf drives.
#
#   app (Xiaomi Pad 6, :benchmark process)
#     └─ native OpenAI server @ 127.0.0.1:8080 (model + GPU layers from Chat tab)
#          │  adb forward tcp:8080 → tcp:8080
#          ▼
#        host aiperf --endpoint-type chat --url http://localhost:8080
#
# The model and the GPU/CPU offload are selected in the app's **Chat tab**
# (Models picker + "GPU offload" toggle + "Start server"). This script verifies
# the server is up, runs the aiperf sweep, and collects the same artifacts
# (report + thermal log) as benchmark-non-reasoning.sh so the chart/report
# tooling is compatible.
#
# Prep (one-time):
#   1. Build + install the app, launch it, open the Chat tab.
#   2. Pick a GGUF model (Models tab downloads; Chat tab selects it).
#   3. Toggle "GPU offload (Vulkan)" for the backend you want to measure.
#   4. Tap "Start server". The card shows "Running · port 8080".
#
# Usage:
#   bash benchmark-pipette-chat.sh                            # current model, CPU label
#   bash benchmark-pipette-chat.sh --backend vulkan --model qwen2.5-0.5b
#   bash benchmark-pipette-chat.sh --backend both --reqs 5
#   bash benchmark-pipette-chat.sh --dry-run
#   bash benchmark-pipette-chat.sh --skip-smoke
#
# Requirements:
#   - adb (device authorised), app installed + chat server started in-app
#   - python3 + aiperf 0.11.0 (uv sync at the clone root → .venv)

set -euo pipefail

# ── Auto-relaunch inside tmux (survives terminal close, like the other benches) ──
if [ -z "${TMUX:-}" ]; then
    SESSION="pipette-chat-bench"
    SELF="$(realpath "$0")"
    ARGS="$(printf '%q ' "$@")"
    ENV_FWDS=""
    [ -n "${AIPERF_BIN:-}" ] && ENV_FWDS+="export AIPERF_BIN=$(printf '%q' "$AIPERF_BIN"); "
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "/opt/homebrew/bin/bash -c '${ENV_FWDS}/opt/homebrew/bin/bash ${SELF} ${ARGS}; echo; echo === Done — press Enter to exit ===; read'"
    echo "Launched in tmux session '$SESSION'."
    echo "Attach with:  tmux attach -t $SESSION"
    exit 0
fi

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
_first_exe() {
    local c
    for c in "$@"; do
        [ -n "${c:-}" ] && [ -x "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
    return 1
}

BACKEND="cpu"        # cpu | vulkan | both — label only; the app's toggle sets it
MODEL_LABEL="pipette-model"
REQS=20
ONLY_COMBO=""
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10
COOLDOWN_MODEL=30
COOLDOWN_BACKEND=45
SERVER_STARTUP_TIMEOUT=120

HOST_PORT=8080
DEVICE_PORT=8080
SERVER_URL="http://localhost:${HOST_PORT}"

CONTEXT_SIZE=2560
PROMPT_LENGTHS=(128 512 1024 2048)
GEN_LENGTHS=(64 128 256)

ADB="adb"
SERIAL=""
AIPERF_BIN="${AIPERF_BIN:-$(_first_exe "$REPO_ROOT/.venv/bin/aiperf" "$REPO_ROOT/venv/bin/aiperf" "$HOME/Desktop/smolbenchmark/venv/bin/aiperf" "$HOME/venv/bin/aiperf" "$(command -v aiperf 2>/dev/null || true)")}"
HF_CLI="${HF_CLI:-$(_first_exe "$REPO_ROOT/.venv/bin/hf" "$REPO_ROOT/venv/bin/hf" "$HOME/Desktop/smolbenchmark/venv/bin/hf" "$(command -v hf 2>/dev/null || true)")}"

BASE_ARTIFACT=""
THERMAL_LOG=""
DEVICE_INFO_FILE=""

# ── Args ──────────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --backend)        BACKEND="$2"; shift 2 ;;
        --model)          MODEL_LABEL="$2"; shift 2 ;;
        --reqs)           REQS="$2"; shift 2 ;;
        --prompt-lengths) PROMPT_LENGTHS=($2); shift 2 ;;
        --gen-lengths)    GEN_LENGTHS=($2); shift 2 ;;
        --host-port)      HOST_PORT="$2"; SERVER_URL="http://localhost:${HOST_PORT}"; shift 2 ;;
        --device-port)    DEVICE_PORT="$2"; shift 2 ;;
        --serial)         SERIAL="$2"; shift 2 ;;
        --skip-smoke)     SKIP_SMOKE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --resume)         RESUME_DIR="$2"; shift 2 ;;
        -h|--help)        sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -n "$SERIAL" ]; then ADB="adb -s $SERIAL"; fi

# ── Resolve artifact dir ──────────────────────────────────────────────────────
if [ -n "$RESUME_DIR" ]; then
    BASE_ARTIFACT="$RESUME_DIR"
    echo "  [RESUME] Reusing artifact dir: $BASE_ARTIFACT"
else
    BASE_ARTIFACT="${SCRIPT_DIR}/artifacts/pipette-chat/${BACKEND}-$(date +%Y%m%d-%H%M)"
fi
THERMAL_LOG="${BASE_ARTIFACT}/thermal.log"
DEVICE_INFO_FILE="${BASE_ARTIFACT}/device-info.txt"

log()  { printf '%s\n' "$*"; }
err()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
ts()   { date '+%Y-%m-%d %H:%M:%S'; }

# ── Prereqs ───────────────────────────────────────────────────────────────────
command -v "${ADB%% *}" >/dev/null || err "adb not found in PATH"
if [ ! -f "$AIPERF_BIN" ] && ! command -v aiperf >/dev/null; then
    err "aiperf not found. From the clone root: uv sync"
fi

DEVICE=$($ADB devices | awk 'NR==2{print $1}')
[ -z "$DEVICE" ] && err "No adb device attached"

echo "=== Pipette chat benchmark ==="
echo "Device:            $DEVICE"
echo "Backend label:     $BACKEND  (app Chat tab must have the matching toggle)"
echo "Model label:       $MODEL_LABEL"
echo "Requests/combo:    $REQS"
echo "Host port:         $HOST_PORT  → device ${DEVICE_PORT}"
echo "Artifact dir:      $BASE_ARTIFACT"

mkdir -p "$BASE_ARTIFACT"
$ADB shell getprop ro.product.model   > "$DEVICE_INFO_FILE" 2>/dev/null || true
$ADB shell getprop ro.hardware.egl   >> "$DEVICE_INFO_FILE" 2>/dev/null || true

# ── Port forward + wait for the in-app server ────────────────────────────────
$ADB forward "tcp:${HOST_PORT}" "tcp:${DEVICE_PORT}" > /dev/null
log "  adb forward tcp:${HOST_PORT} → device:${DEVICE_PORT}"

wait_for_server() {
    log "  Waiting for the in-app server (start it in the Chat tab)…"
    for i in $(seq 1 "$SERVER_STARTUP_TIMEOUT"); do
        code=$(curl -s "${SERVER_URL}/v1/models" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")
        [ "$code" = "200" ] && { log "  Server up after ${i}s"; return 0; }
        sleep 1
    done
    err "Server did not come up on ${SERVER_URL}. Open the app → Chat tab → pick model → Start server."
}
if [ "$DRY_RUN" = "0" ]; then wait_for_server; fi

# ── On-device labeled SoC thermal / cpufreq / RSS sampler (whole run) ─────────
# Reads Qualcomm tsens zone TYPES ON THE DEVICE (unrooted; only the battery
# *power* rail is Permission-denied — not needed) and streams CSV lines back
# over adb into thermal.log. On-device loop (pushed script) removes the
# host-adb round-trip so sampling can reach Jetson-grade density
# (tegrastats --interval 500 ⇒ THERMAL_INTERVAL=0.5). Set THERMAL_INTERVAL.
# Columns: ts_epoch_s,cpu_max_c,gpu_c,npu_c,ddr_c,pmic_c,batt_c,cpu_max_mhz,rss_mb
APP_PKG="${APP_PKG:-ai.liquid.pipette.debug}"
THERMAL_INTERVAL="${THERMAL_INTERVAL:-1}"  # s between SoC samples; 0.5 = Jetson tegrastats parity
SAMPLER_PIDFILE="/tmp/pipette_bench_thermal.pid"
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
    $ADB shell "nohup $DEVICE_POLL $THERMAL_INTERVAL '$APP_PKG' $DEVICE_OUT >/dev/null 2>&1 &"
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
        log "  [DRY] aiperf --endpoint-type chat --url ${SERVER_URL} ctx=${ct} gen=${gen} reqs=${REQS}"
        return 0
    fi

    # Smoke check: one real completion before the sweep (skip with --skip-smoke)
    if [ "$SKIP_SMOKE" = "0" ]; then
        resp=$(curl -s "${SERVER_URL}/v1/chat/completions" \
            --max-time "$REQUEST_TIMEOUT" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${MODEL_LABEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":16}" || true)
        echo "$resp" | grep -q '"choices"' || log "  [SMOKE] no completion payload (continuing anyway)"
        log "  [SMOKE PASS] ${MODEL_LABEL} ctx=${ct} gen=${gen}"
    fi

    log "  aiperf sweep ctx=${ct} gen=${gen} reqs=${REQS}"
    "$AIPERF_BIN" \
        --endpoint-type 'chat' \
        --url          "${SERVER_URL}" \
        --model        "${MODEL_LABEL}" \
        --prompt       "$ct" \
        --max-tokens   "$gen" \
        --num-requests "$REQS" \
        --artifact-dir "$combo_dir" \
        || log "  aiperf failed (ctx=${ct} gen=${gen})"
    sleep "$COOLDOWN_COMBO"
}

if [ "$DRY_RUN" = "0" ]; then start_sampler; fi
for ct in "${PROMPT_LENGTHS[@]}"; do
    for gen in "${GEN_LENGTHS[@]}"; do
        if [ -n "$ONLY_COMBO" ] && [ "$ONLY_COMBO" != "${ct}:${gen}" ]; then continue; fi
        run_combo "$ct" "$gen"
    done
done
stop_sampler

echo
echo "=== Done. Artifacts in ${BASE_ARTIFACT} ==="
echo "    thermal.log:  $(tail -1 "$THERMAL_LOG" 2>/dev/null)"
echo "    aiperf plot ${BASE_ARTIFACT} --dashboard --port 8050"
