#!/opt/homebrew/bin/bash
# benchmark_android.sh
#
# Benchmarks tiny LLMs on an Android phone (USB-connected) using:
#   - llama-server (arm64 binary pushed to /data/local/tmp) as the inference backend
#   - adb forward  to tunnel localhost:PORT → device:PORT
#   - aiperf       (from Mac/Linux host) as the load generator — same as Jetson
#   - adb shell    polling /sys/class/thermal/* for real °C per iteration
#
# Supports two backends on the same run:
#   CPU  (-ngl 0)  — pure Cortex-A78/A55, always works
#   GPU  (-ngl 99) — Mali-G68 via Vulkan, requires Vulkan build of llama-server
#
# Mirrors the structure of benchmark_all_bonsai.sh / bench-non-reasoning.sh
# so artifacts, report format, and chart scripts are compatible.
#
# Phone requirements:
#   - Android 10+ (API 29+), USB debugging enabled
#   - ADB authorised (trust prompt accepted on device)
#   - llama-server CPU binary at /data/local/tmp/llama-server
#   - llama-server Vulkan binary at /data/local/tmp/llama-server-vulkan (for --backend vulkan/both)
#   - GGUF models pushed to /data/local/tmp/models/ OR auto-pulled via hf
#
# Host requirements:
#   - adb in PATH
#   - python3 + pip install aiperf
#   - hf CLI (pip install huggingface_hub[cli]) for model downloads
#
# Usage:
#   bash benchmark_android.sh                              # all models, CPU only
#   bash benchmark_android.sh --backend vulkan             # GPU (Vulkan) only
#   bash benchmark_android.sh --backend both               # CPU then GPU — full comparison
#   bash benchmark_android.sh --backend both --only qwen2.5-0.5b --reqs 5
#   bash benchmark_android.sh --skip-smoke
#   bash benchmark_android.sh --dry-run
#   bash benchmark_android.sh --resume artifacts/android/run-20260616-1200
#   bash benchmark_android.sh --prompt-lengths "64 256 512"
#   bash benchmark_android.sh --gen-lengths "64 128 256"

set -euo pipefail

# ── Auto-relaunch inside tmux ─────────────────────────────────────────────────
if [ -z "${TMUX:-}" ]; then
    SESSION="android-bench"
    SELF="$(realpath "$0")"
    ARGS="$(printf '%q ' "$@")"
    # Forward required env vars into the tmux session; use bash 5 (macOS ships 3.2)
    ENV_FWDS=""
    [ -n "${AIPERF_BIN:-}" ] && ENV_FWDS+="export AIPERF_BIN=$(printf '%q' "$AIPERF_BIN"); "
    [ -n "${HF_CLI:-}" ]     && ENV_FWDS+="export HF_CLI=$(printf '%q' "$HF_CLI"); "
    [ -n "${HF_TOKEN:-}" ]   && ENV_FWDS+="export HF_TOKEN=$(printf '%q' "$HF_TOKEN"); "
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "/opt/homebrew/bin/bash -c '${ENV_FWDS}/opt/homebrew/bin/bash ${SELF} ${ARGS}; echo; echo === Done — press Enter to exit ===; read'"
    echo "Launched in tmux session '$SESSION'."
    echo "Attach with:  tmux attach -t $SESSION"
    exit 0
fi

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backend: cpu | vulkan | both
# cpu    → llama-server -ngl 0  (pure Cortex-A78/A55, always works)
# vulkan → llama-server -ngl 99 (Mali-G68 via Vulkan, needs Vulkan build)
# both   → runs cpu first, then vulkan, artifacts split by backend
BACKEND="cpu"

REQS=20
ONLY_MODEL=""
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
CONCURRENCY=1
SLICE_DURATION=30
RANDOM_SEED=42
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10             # seconds between prompt×gen combos
COOLDOWN_MODEL=30             # seconds between models (thermal cooldown)
COOLDOWN_BACKEND=45           # seconds between cpu→vulkan switch (thermal reset)
SERVER_STARTUP_TIMEOUT=120    # seconds to wait for llama-server HTTP 200

HOST_PORT=8080
DEVICE_PORT=8080
SERVER_URL="http://localhost:${HOST_PORT}"

CONTEXT_SIZE=2560   # max_prompt(2048) + max_gen(256) = 2304, padded to 2560

# Sweep defaults — override with --prompt-lengths / --gen-lengths
PROMPT_LENGTHS=(128 512 1024 2048)
GEN_LENGTHS=(64 128 256)

# Device paths
DEVICE_TMP="/data/local/tmp"
DEVICE_MODEL_DIR="${DEVICE_TMP}/models"
DEVICE_BIN_CPU="${DEVICE_TMP}/llama-server"         # CPU-only build (always required)
DEVICE_BIN_VULKAN="${DEVICE_TMP}/llama-server-vulkan" # Vulkan build (required for --backend vulkan/both)

# Active binary — set per backend pass in the run loop
DEVICE_BIN="${DEVICE_BIN_CPU}"

# Host paths
HF_CLI="${HF_CLI:-$(which hf 2>/dev/null || echo "$HOME/.local/bin/hf")}"
AIPERF_BIN="${AIPERF_BIN:-$(which aiperf 2>/dev/null || echo "$HOME/venv/bin/aiperf")}"

# Artifact dirs and logfiles — resolved after arg parsing
BASE_ARTIFACT=""
THERMAL_LOG=""
TIMING_LOG=""
DEVICE_INFO_FILE=""

SERVER_PIDFILE="/tmp/android_bench_server.pid"   # PID of adb shell llama-server
THERMAL_PIDFILE="/tmp/android_bench_thermal.pid"  # PID of thermal polling loop

# ── Model table ───────────────────────────────────────────────────────────────
# Format: "display_name|quant|device_gguf_path|hf_tokenizer|context_size"
# device_gguf_path uses $GGUF_DIR (flat — no family subdirs).
GGUF_DIR="${DEVICE_MODEL_DIR}"
HOST_GGUF_DIR="/tmp/smolbench_models"

declare -a MODELS=(
    "lfm2.5-1.2b|Q4_K_M|$GGUF_DIR/lfm2.5-1.2b-q4_k_m.gguf|LiquidAI/LFM2.5-1.2B-Instruct|2560"
    "llama3.2-1b|Q4_K_M|$GGUF_DIR/llama3-2-1bq4_k_m.gguf|meta-llama/Llama-3.2-1B-Instruct|2560"
    "gemma3-1b|Q4_K_M|$GGUF_DIR/gemma3-1b-q4_k_m.gguf|google/gemma-3-1b-it|2560"
    "qwen3-0.6b|Q8_0|$GGUF_DIR/qwen3-0-6bq8_0.gguf|Qwen/Qwen3-0.6B|2560"
    "qwen2.5-0.5b|Q4_K_M|$GGUF_DIR/qwen2-5-0-5bq4_k_m.gguf|Qwen/Qwen2.5-0.5B-Instruct|2560"
    "smollm2-360m|Q8_0|$GGUF_DIR/smollm2-360mq8_0.gguf|HuggingFaceTB/SmolLM2-360M-Instruct|2560"
    "lfm2.5-350m|Q4_K_M|$GGUF_DIR/lfm2.5-350m-q4_k_m.gguf|LiquidAI/LFM2.5-350M|2560"
    "smollm2-135m|Q4_K_M|$GGUF_DIR/smollm2-135mq4_k_m.gguf|HuggingFaceTB/SmolLM2-135M-Instruct|2560"
)

# ── GGUF download sources: local_filename -> "hf_repo hf_filename" ────────────
declare -A GGUF_SOURCES=(
    ["smollm2-135mq4_k_m.gguf"]="bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf"
    ["smollm2-360mq8_0.gguf"]="bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q8_0.gguf"
    ["qwen2-5-0-5bq4_k_m.gguf"]="Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ["qwen3-0-6bq8_0.gguf"]="Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf"
    ["llama3-2-1bq4_k_m.gguf"]="bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["gemma3-1b-q4_k_m.gguf"]="lmstudio-community/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf"
    ["gemma3-4b-q4_k_m.gguf"]="lmstudio-community/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf"
    ["lfm2.5-350m-q4_k_m.gguf"]="LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf"
    ["lfm2.5-1.2b-q4_k_m.gguf"]="LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)        BACKEND="$2";                            shift 2 ;;
        --reqs)           REQS="$2";                               shift 2 ;;
        --only)           ONLY_MODEL="$2";                         shift 2 ;;
        --skip-smoke)     SKIP_SMOKE=1;                            shift   ;;
        --dry-run)        DRY_RUN=1;                               shift   ;;
        --resume)         RESUME_DIR="$2";                         shift 2 ;;
        --prompt-lengths) read -ra PROMPT_LENGTHS <<< "$2";        shift 2 ;;
        --gen-lengths)    read -ra GEN_LENGTHS    <<< "$2";        shift 2 ;;
        --host-port)      HOST_PORT="$2"; SERVER_URL="http://localhost:${HOST_PORT}"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$BACKEND" in
    cpu|vulkan|both) ;;
    *) echo "ERROR: --backend must be cpu, vulkan, or both"; exit 1 ;;
esac

# ── Resolve artifact dir ──────────────────────────────────────────────────────
if [ -n "$RESUME_DIR" ]; then
    # Resolve relative paths against SCRIPT_DIR so --resume works from any cwd (e.g. tmux)
    [[ "$RESUME_DIR" != /* ]] && RESUME_DIR="${SCRIPT_DIR}/${RESUME_DIR}"
    [ -d "$RESUME_DIR" ] || { echo "ERROR: --resume dir not found: $RESUME_DIR"; exit 1; }
    BASE_ARTIFACT="$RESUME_DIR"
    echo "  [RESUME] Reusing artifact dir: $BASE_ARTIFACT"
else
    BASE_ARTIFACT="${SCRIPT_DIR}/artifacts/android/${BACKEND}-$(date +%Y%m%d-%H%M)"
fi

THERMAL_LOG="${BASE_ARTIFACT}/thermal.log"
CPUFREQ_LOG="${BASE_ARTIFACT}/cpufreq.log"
BATTERY_LOG="${BASE_ARTIFACT}/battery.log"
TIMING_LOG="${BASE_ARTIFACT}/model_timing.log"
DEVICE_INFO_FILE="${BASE_ARTIFACT}/device_info.txt"

CPUFREQ_PIDFILE="/tmp/android_bench_cpufreq.pid"
BATTERY_PIDFILE="/tmp/android_bench_battery.pid"

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
log() { echo "  [$(date +%H:%M:%S)] $*"; }

# ── ADB helpers ───────────────────────────────────────────────────────────────
adb_check() {
    # If ANDROID_SERIAL is already set (e.g. WiFi ADB), just verify that specific device
    if [ -n "${ANDROID_SERIAL:-}" ]; then
        if adb -s "$ANDROID_SERIAL" shell echo ok >/dev/null 2>&1; then
            log "ADB device: ${ANDROID_SERIAL}"
            return 0
        else
            echo "ERROR: Device $ANDROID_SERIAL not reachable."
            exit 1
        fi
    fi
    local all_devices usb_serial
    # Prefer USB transport (no dots in serial) when multiple devices are connected
    usb_serial=$(adb devices 2>/dev/null | grep -v "^List" | awk '/device$/ && $1 !~ /\./{print $1; exit}')
    all_devices=$(adb devices 2>/dev/null | grep -v "^List" | grep -c "device$" || true)
    if [ "${all_devices:-0}" -eq 0 ]; then
        echo "ERROR: No authorised Android device found. Check USB cable and trust prompt."
        exit 1
    fi
    if [ -n "$usb_serial" ] && [ "${all_devices:-0}" -gt 1 ]; then
        export ANDROID_SERIAL="$usb_serial"
        log "Multiple devices detected — auto-selected USB serial: ${ANDROID_SERIAL}"
    fi
}

device_model() {
    adb shell getprop ro.product.model 2>/dev/null | tr -d '\r'
}

device_soc() {
    adb shell getprop ro.board.platform 2>/dev/null | tr -d '\r'
}

device_android_version() {
    adb shell getprop ro.build.version.release 2>/dev/null | tr -d '\r'
}

device_cpu_topology() {
    # Returns big.LITTLE core count from /proc/cpuinfo (number of processors)
    adb shell "grep -c '^processor' /proc/cpuinfo" 2>/dev/null | tr -d '\r'
}

device_cpu_max_freqs() {
    # Max freq per core from cpufreq (kHz), one per line
    adb shell "cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq 2>/dev/null" \
        | tr -d '\r' | sort -u
}

device_cpu_arch() {
    adb shell "grep 'CPU architecture' /proc/cpuinfo" 2>/dev/null \
        | head -1 | awk -F: '{print $2}' | tr -d ' \r'
}

device_total_ram_mb() {
    adb shell cat /proc/meminfo 2>/dev/null \
        | awk '/MemTotal/{print int($2/1024)}' | tr -d '\r'
}

# ── Thermal zone discovery ────────────────────────────────────────────────────
# Primary path: dumpsys thermalservice HAL (works on ColorOS where
# /sys/class/thermal/*/type is permission-denied).
# Discovers thermal zones via Android thermal HAL (dumpsys thermalservice).
# Output format per line: key:Name:DUMPSYS
discover_thermal_zones() {
    local hal_zones
    hal_zones=$(adb shell "dumpsys thermalservice 2>/dev/null" | tr -d '\r' | awk '
        /^Current temperatures from HAL/{p=1;next}
        p && /^[A-Z]/{p=0}
        p && /mName=/{
            n=$0; gsub(/.*mName=/,"",n); gsub(/[,}].*/,"",n)
            print n
        }
    ')

    if [ -z "$hal_zones" ]; then
        log "  [THERMAL WARN] No thermal zones from HAL — thermal data unavailable"
        return
    fi

    while IFS= read -r name; do
        [ -z "$name" ] && continue
        local key
        key=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr -d '_')
        echo "${key}:${name}:DUMPSYS"
    done <<< "$hal_zones"
}

# ── Thermal poller ────────────────────────────────────────────────────────────
# Polls via dumpsys thermalservice (Android thermal HAL).
# Pushes a self-contained poll script to the device to avoid quoting issues.
start_thermal_poller() {
    local zone_spec="$1"

    local zone_names=()
    local headers="timestamp_ms"
    while IFS= read -r entry; do
        [ -z "$entry" ] && continue
        local key="${entry%%:*}"
        local rest="${entry#*:}"
        local name="${rest%%:*}"
        zone_names+=("$name")
        headers="${headers},${key}_degC"
    done <<< "$zone_spec"

    echo "$headers" > "$THERMAL_LOG"

    local zones_line="${zone_names[*]}"
    local tmp_poll
    tmp_poll=$(mktemp /tmp/thermal_poll_XXXXX.sh)

    # Variables escaped with \ run on device; ${zones_line} expands locally now.
    cat > "$tmp_poll" << POLLSCRIPT
#!/system/bin/sh
ZONES="${zones_line}"
while true; do
  ts=\$(date +%s%3N)
  raw=\$(dumpsys thermalservice 2>/dev/null)
  row="\$ts"
  for name in \$ZONES; do
    v=\$(printf '%s\n' "\$raw" | awk -v z="\$name" '
      /^Current temperatures from HAL/{p=1;next}
      p && /^[A-Z]/{p=0}
      p && /mName=/{
        n=\$0; gsub(/.*mName=/,"",n); gsub(/[,}].*/,"",n)
        if (n==z){v=\$0; gsub(/.*mValue=/,"",v); gsub(/,.*/,"",v); print v+0; exit}
      }
    ')
    row="\${row},\${v:-0}"
  done
  echo "\$row"
  sleep 0.5
done
POLLSCRIPT

    adb push "$tmp_poll" /data/local/tmp/thermal_poll.sh >/dev/null 2>&1
    adb shell chmod +x /data/local/tmp/thermal_poll.sh
    rm -f "$tmp_poll"

    adb shell /data/local/tmp/thermal_poll.sh >> "$THERMAL_LOG" &
    echo $! > "$THERMAL_PIDFILE"
    log "  Thermal poller PID $(cat $THERMAL_PIDFILE)  → $THERMAL_LOG"
    log "  Source: dumpsys thermalservice (HAL) — zones: ${zones_line}"
    log "  Columns: $headers"
}

stop_thermal_poller() {
    if [ -f "$THERMAL_PIDFILE" ]; then
        kill "$(cat "$THERMAL_PIDFILE")" 2>/dev/null || true
        rm -f "$THERMAL_PIDFILE"
    fi
    pkill -f "adb shell /data/local/tmp/thermal_poll.sh" 2>/dev/null || true
    adb shell "pkill -f thermal_poll.sh" 2>/dev/null || true
}

# ── CPU frequency poller ──────────────────────────────────────────────────────
# Polls scaling_cur_freq for all cores every 500ms.
# Dimensity 7050 (MT6877) layout:
#   cpu0, cpu1  → Cortex-A78 @ up to 2.6 GHz  (big cores, performance)
#   cpu2..cpu7  → Cortex-A55 @ up to 2.0 GHz  (little cores, efficiency)
# Per arXiv:2410.03613: optimal thread count = number of big cores (2 here).
# Output: CSV → CPUFREQ_LOG
# Header: timestamp_ms,cpu0_kHz,cpu1_kHz,...,cpu7_kHz
start_cpufreq_poller() {
    log "  Starting CPU frequency poller..."

    # Discover how many cores exist
    local num_cores
    num_cores=$(adb shell "ls /sys/devices/system/cpu/ 2>/dev/null | grep -c '^cpu[0-9]'" | tr -d '\r')
    num_cores=${num_cores:-8}

    # Build header
    local header="timestamp_ms"
    for i in $(seq 0 $((num_cores - 1))); do
        header="${header},cpu${i}_kHz"
    done
    echo "$header" > "$CPUFREQ_LOG"

    # Build polling command
    local read_cmd="while true; do
        ts=\$(date +%s%3N)
        row=\"\$ts\""
    for i in $(seq 0 $((num_cores - 1))); do
        read_cmd+="
        f=\$(cat /sys/devices/system/cpu/cpu${i}/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
        row=\"\${row},\${f}\""
    done
    read_cmd+="
        echo \"\$row\"
        sleep 0.5
    done"

    adb shell "$read_cmd" >> "$CPUFREQ_LOG" &
    echo $! > "$CPUFREQ_PIDFILE"
    log "  CPU freq poller PID $(cat $CPUFREQ_PIDFILE)  → $CPUFREQ_LOG"
    log "  Columns: $header"
}

stop_cpufreq_poller() {
    if [ -f "$CPUFREQ_PIDFILE" ]; then
        kill "$(cat "$CPUFREQ_PIDFILE")" 2>/dev/null || true
        rm -f "$CPUFREQ_PIDFILE"
    fi
    pkill -f "adb shell.*scaling_cur_freq" 2>/dev/null || true
}

# ── Battery / power poller ────────────────────────────────────────────────────
# Polls every 2s via sysfs. current_now (µA) × voltage_now (µV) = µW → W.
# current_now is 0 when USB charging matches load; goes negative (discharge)
# under heavy inference that exceeds USB power budget — that's when we get real watts.
# Output: CSV → BATTERY_LOG
# Header: timestamp_ms,level_pct,current_ua,voltage_uv,power_mw,temp_tenths_degC,status
start_battery_poller() {
    log "  Starting battery/power poller (2s interval)..."

    echo "timestamp_ms,level_pct,current_ua,voltage_uv,power_mw,temp_tenths_degC,status" > "$BATTERY_LOG"

    local read_cmd="while true; do
        ts=\$(date +%s%3N)
        cur=\$(cat /sys/class/power_supply/battery/current_now 2>/dev/null || echo 0)
        vlt=\$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null || echo 0)
        dump=\$(dumpsys battery 2>/dev/null)
        level=\$(echo \"\$dump\" | awk '/  level:/{print \$2}' | tr -d '\r')
        temp=\$(echo \"\$dump\"  | awk '/temperature:/{print \$2}' | tr -d '\r')
        stat=\$(echo \"\$dump\"  | awk '/  status:/{print \$2}' | tr -d '\r')
        pw=\$(awk -v c=\"\${cur:-0}\" -v v=\"\${vlt:-0}\" 'BEGIN{printf \"%.1f\", (c<0 ? -c : c)*v/1000000000}')
        echo \"\${ts},\${level:-0},\${cur:-0},\${vlt:-0},\${pw},\${temp:-0},\${stat:-0}\"
        sleep 2
    done"

    adb shell "$read_cmd" >> "$BATTERY_LOG" &
    echo $! > "$BATTERY_PIDFILE"
    log "  Battery/power poller PID $(cat $BATTERY_PIDFILE)  → $BATTERY_LOG"
    log "  Columns: timestamp_ms,level_pct,current_ua,voltage_uv,power_mw,temp_tenths_degC,status"
}

stop_battery_poller() {
    if [ -f "$BATTERY_PIDFILE" ]; then
        kill "$(cat "$BATTERY_PIDFILE")" 2>/dev/null || true
        rm -f "$BATTERY_PIDFILE"
    fi
    pkill -f "adb shell.*dumpsys battery" 2>/dev/null || true
}

# ── ADB port forward ──────────────────────────────────────────────────────────
setup_port_forward() {
    adb forward "tcp:${HOST_PORT}" "tcp:${DEVICE_PORT}" > /dev/null
    log "  adb forward tcp:${HOST_PORT} → device:${DEVICE_PORT}"
}

teardown_port_forward() {
    adb forward --remove "tcp:${HOST_PORT}" 2>/dev/null || true
}

# ── Server management ─────────────────────────────────────────────────────────
start_server() {
    local model_path="$1"   # path ON device
    local ctx_size="$2"
    local model_name="$3"
    local run_backend="${4:-cpu}"   # cpu | vulkan
    local srv_log="${BASE_ARTIFACT}/${model_name}-${run_backend}-server.log"

    local bin ngl_flag
    if [ "$run_backend" = "vulkan" ]; then
        bin="$DEVICE_BIN_VULKAN"
        ngl_flag="-ngl 99"   # offload all layers to Mali-G68 via Vulkan
    else
        bin="$DEVICE_BIN_CPU"
        ngl_flag="-ngl 0"    # CPU only — Cortex-A78/A55
    fi

    log "  Starting llama-server [${run_backend}] on device (port ${DEVICE_PORT})..."
    log "  Binary: ${bin}  |  ngl: ${ngl_flag}"

    # Kill any existing server on device
    adb shell "pkill -f llama-server" 2>/dev/null || true
    sleep 2

    # Launch server on device in background.
    # --no-cache-prompt ensures fair TTFT every request (no KV cache reuse).
    # taskset 0xFF pins to all 8 cores (cpu0-3=A55 + cpu4-7=A78).
    # -t 8 uses all cores — better TTFT (prefill is compute-heavy); decode is memory-bound anyway.
    adb shell "LD_LIBRARY_PATH=${DEVICE_TMP} \
        taskset ff \
        ${bin} \
        -m ${model_path} \
        --host 127.0.0.1 \
        --port ${DEVICE_PORT} \
        --parallel 1 \
        -c ${ctx_size} \
        -t 8 \
        ${ngl_flag} \
        --no-cache-prompt \
        > ${DEVICE_TMP}/llama-server-${run_backend}.log 2>&1 &
        echo \$!" > "$SERVER_PIDFILE" &

    sleep 3

    log "  Waiting for HTTP 200 (timeout: ${SERVER_STARTUP_TIMEOUT}s)..."
    local elapsed=0
    local code
    while [ "$elapsed" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
        sleep 2; elapsed=$((elapsed + 2))
        code=$(curl -s "${SERVER_URL}/v1/models" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")
        if   [ "$code" = "200" ]; then
            log "  [OK] HTTP 200 at t=${elapsed}s  [${run_backend}]"
            echo "$srv_log" >> "${BASE_ARTIFACT}/server_logs.txt"
            return 0
        elif [ "$code" = "503" ]; then
            log "  t=${elapsed}s: loading weights..."
        else
            log "  t=${elapsed}s: HTTP $code"
        fi
    done

    log "  [FAIL] Server did not respond within ${SERVER_STARTUP_TIMEOUT}s"
    adb shell cat "${DEVICE_TMP}/llama-server-${run_backend}.log" 2>/dev/null | tail -20 | sed 's/^/      /' || true
    return 1
}

kill_server() {
    adb shell "pkill -f llama-server" 2>/dev/null || true
    teardown_port_forward
    rm -f "$SERVER_PIDFILE"
    sleep 5
}

ensure_server_alive() {
    local model_path="$1"
    local ctx_size="$2"
    local model_name="$3"
    local run_backend="${4:-cpu}"
    local code
    code=$(curl -s "${SERVER_URL}/v1/models" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")
    [ "$code" = "200" ] && return 0
    log "  [!] Server not responding (HTTP $code) — restarting [${run_backend}]..."
    kill_server
    setup_port_forward
    start_server "$model_path" "$ctx_size" "$model_name" "$run_backend"
}

# ── Model download ────────────────────────────────────────────────────────────
# Downloads GGUF to host then pushes to device via adb push.
# Skips if already present on device.
ensure_model_on_device() {
    local model_name="$1"
    local device_path="$2"

    # Check if model already on device
    local exists
    exists=$(adb shell "[ -f '${device_path}' ] && echo yes || echo no" 2>/dev/null | tr -d '\r')
    if [ "$exists" = "yes" ]; then
        log "  ${model_name}: already on device at ${device_path}"
        return 0
    fi

    # Look up download source from GGUF_SOURCES
    local fname
    fname=$(basename "$device_path")
    local src="${GGUF_SOURCES[$fname]:-}"
    if [ -z "$src" ]; then
        log "  [FAIL] No GGUF_SOURCES entry for ${fname} — cannot download ${model_name}"
        return 1
    fi
    local hf_repo="${src%% *}"
    local hf_file="${src##* }"

    # Download to host flat dir
    mkdir -p "$HOST_GGUF_DIR"
    local host_path="${HOST_GGUF_DIR}/${fname}"

    if [ ! -f "$host_path" ]; then
        log "  Downloading ${model_name} from ${hf_repo}..."
        "$HF_CLI" download "$hf_repo" "$hf_file" --local-dir "$HOST_GGUF_DIR" || {
            log "  [FAIL] Download failed for ${model_name}"
            return 1
        }
        # hf download may put it in a subfolder — move it to flat dir if needed
        local downloaded
        downloaded=$(find "$HOST_GGUF_DIR" -name "$hf_file" ! -path "$host_path" 2>/dev/null | head -1)
        if [ -n "$downloaded" ] && [ ! -f "$host_path" ]; then
            mv "$downloaded" "$host_path"
        fi
    fi

    # Push to device
    adb shell "mkdir -p $(dirname "$device_path")" 2>/dev/null || true
    log "  Pushing ${model_name} to device ($(du -sh "$host_path" | cut -f1))..."
    adb push "$host_path" "$device_path"
    log "  [OK] ${model_name} on device at ${device_path}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

# Trap: always restore device state on exit/crash so airplane mode never gets stuck
_restore_device() {
    adb ${ANDROID_SERIAL:+-s "$ANDROID_SERIAL"} shell svc data enable 2>/dev/null || true
    adb ${ANDROID_SERIAL:+-s "$ANDROID_SERIAL"} shell cmd power set-fixed-performance-mode-enabled false 2>/dev/null || true
    adb ${ANDROID_SERIAL:+-s "$ANDROID_SERIAL"} shell svc power stayon false 2>/dev/null || true
    adb ${ANDROID_SERIAL:+-s "$ANDROID_SERIAL"} shell input keyevent 26 2>/dev/null || true
}
trap '_restore_device' EXIT

banner "Android LLM Benchmark  |  smolbench-mobile"

# Phase 0: preflight checks
banner "Phase 0: Preflight"

[ "$DRY_RUN" = 0 ] && adb_check

# ── WiFi ADB setup ────────────────────────────────────────────────────────────
# Switch to ADB over WiFi so USB can be unplugged during the benchmark.
# Unplugged USB = battery discharges freely = real power_mw readings in battery.csv.
# If already on WiFi transport (ANDROID_SERIAL contains a dot), skip this block.
if [ "$DRY_RUN" = 0 ] && [[ "${ANDROID_SERIAL:-}" != *.* ]]; then
    PHONE_IP=$(adb ${ANDROID_SERIAL:+-s "$ANDROID_SERIAL"} shell "ip addr show wlan0 2>/dev/null" | tr -d '\r' | awk '/inet /{print $2}' | cut -d/ -f1 | head -1)
    if [ -n "$PHONE_IP" ]; then
        log "Setting up WiFi ADB (phone IP: ${PHONE_IP})..."
        adb ${ANDROID_SERIAL:+-s "$ANDROID_SERIAL"} tcpip 5555 2>/dev/null || true
        sleep 2
        adb connect "${PHONE_IP}:5555" 2>/dev/null || true
        sleep 2
        if adb -s "${PHONE_IP}:5555" shell echo ok 2>/dev/null | grep -q "^ok"; then
            export ANDROID_SERIAL="${PHONE_IP}:5555"
            log "  [OK] WiFi ADB active — ANDROID_SERIAL=${ANDROID_SERIAL}"
            echo ""
            echo "  ┌──────────────────────────────────────────────────────┐"
            echo "  │  >>> UNPLUG THE USB CABLE NOW <<<                    │"
            echo "  │  ADB is connected over WiFi (${PHONE_IP}:5555)  │"
            echo "  │  Unplugging allows real battery discharge readings.   │"
            echo "  └──────────────────────────────────────────────────────┘"
            echo ""
            read -r -p "  Press Enter once USB is unplugged to continue... " _
            log "  Verifying WiFi ADB still reachable..."
            local retries=3
            while [ $retries -gt 0 ]; do
                if adb -s "$ANDROID_SERIAL" shell echo ok 2>/dev/null | grep -q "^ok"; then
                    log "  [OK] WiFi ADB confirmed — ${ANDROID_SERIAL}"
                    break
                fi
                retries=$((retries - 1))
                if [ $retries -eq 0 ]; then
                    echo "  [ERROR] WiFi ADB lost after USB unplug — is phone on same WiFi?"
                    echo "  Reconnect USB and re-run, or ensure phone WiFi is active."
                    exit 1
                fi
                log "  [WARN] WiFi ADB not responding, retrying in 3s... (${retries} left)"
                sleep 3
            done
        else
            log "  [WARN] WiFi ADB connect failed — staying on USB transport"
            log "         (power_mw readings may be 0 if battery is full while charging)"
        fi
    else
        log "  [WARN] Could not get phone WiFi IP — is WiFi enabled on the phone?"
        log "         Staying on USB transport; power_mw may read 0 at full charge"
    fi
fi

if [ ! -f "$AIPERF_BIN" ] && ! command -v aiperf &>/dev/null; then
    echo "ERROR: aiperf not found. Install with: pip install aiperf"
    exit 1
fi

if [ "$DRY_RUN" = 0 ] && ! command -v hf &>/dev/null && [ ! -f "$HF_CLI" ]; then
    echo "ERROR: hf CLI not found. Install with: pip install 'huggingface_hub[cli]'"
    exit 1
fi

mkdir -p "$BASE_ARTIFACT"

# Reduce power noise and keep phone awake during benchmark.
if [ "$DRY_RUN" = 0 ]; then
    log "Keep screen/WiFi awake during benchmark (prevents WiFi ADB drop)..."
    adb shell svc power stayon true 2>/dev/null || true

    log "Screen off (reduces display draw ~300-600mW)..."
    adb shell input keyevent 26 2>/dev/null || true

    log "Disabling mobile data (reduces modem draw ~200-500mW)..."
    adb shell svc data disable 2>/dev/null || true

    log "Requesting Android performance mode..."
    adb shell cmd power set-fixed-performance-mode-enabled true 2>/dev/null || true
fi

# Collect device info
if [ "$DRY_RUN" = 0 ]; then
    log "Collecting device info..."
    {
        echo "Model:           $(device_model)"
        echo "SoC:             $(device_soc)"
        echo "Android version: $(device_android_version)"
        echo "Total RAM:       $(device_total_ram_mb) MB"
        echo "CPU cores:       $(device_cpu_topology)"
        echo "CPU arch:        $(device_cpu_arch)"
        echo "CPU max freqs (kHz, unique):"
        device_cpu_max_freqs | sed 's/^/  /'
        echo "ADB serial:      ${ANDROID_SERIAL:-<default>}"
        echo "Benchmark date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Backend:         ${BACKEND}"
        echo "CPU binary:      ${DEVICE_BIN_CPU}"
        echo "Vulkan binary:   ${DEVICE_BIN_VULKAN}"
        echo "Host port:       ${HOST_PORT}"
        echo "Device port:     ${DEVICE_PORT}"
        echo "Requests/combo:  ${REQS}"
        echo "Prompt lengths:  ${PROMPT_LENGTHS[*]}"
        echo "Gen lengths:     ${GEN_LENGTHS[*]}"
        echo "Context size:    ${CONTEXT_SIZE}"
        echo ""
        echo "--- /proc/cpuinfo ---"
        adb shell cat /proc/cpuinfo 2>/dev/null | tr -d '\r' \
            | grep -E "^(processor|model name|Hardware|CPU part|CPU architecture|CPU implementer)" \
            | head -40
    } | tee "$DEVICE_INFO_FILE"
fi

echo ""
echo "  Backend    : ${BACKEND}"
echo "  Requests   : ${REQS} per combo"
echo "  Prompts    : ${PROMPT_LENGTHS[*]}"
echo "  Gen lens   : ${GEN_LENGTHS[*]}"
echo "  Artifacts  : ${BASE_ARTIFACT}"
[ -n "$ONLY_MODEL" ] && echo "  Filter     : ${ONLY_MODEL}"
[ "$DRY_RUN" = 1 ]   && echo "  Mode       : DRY RUN"
[ -n "$RESUME_DIR" ]  && echo "  Mode       : RESUME"

# Check required binaries on device
if [ "$DRY_RUN" = 0 ]; then
    # CPU binary is always required
    cpu_ok=$(adb shell "[ -f '${DEVICE_BIN_CPU}' ] && echo yes || echo no" 2>/dev/null | tr -d '\r')
    if [ "$cpu_ok" != "yes" ]; then
        echo ""
        echo "ERROR: CPU llama-server binary not found at ${DEVICE_BIN_CPU}"
        echo "  Build (CPU only):"
        echo "    cmake -B build-android \\"
        echo "        -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK/build/cmake/android.toolchain.cmake \\"
        echo "        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DGGML_OPENMP=OFF"
        echo "    cmake --build build-android --target llama-server -j\$(nproc)"
        echo "    adb push build-android/bin/llama-server ${DEVICE_BIN_CPU}"
        echo "    adb shell chmod +x ${DEVICE_BIN_CPU}"
        exit 1
    fi
    log "CPU binary OK: ${DEVICE_BIN_CPU}"

    # Vulkan binary check — only required if --backend vulkan or both
    if [ "$BACKEND" = "vulkan" ] || [ "$BACKEND" = "both" ]; then
        vulkan_ok=$(adb shell "[ -f '${DEVICE_BIN_VULKAN}' ] && echo yes || echo no" 2>/dev/null | tr -d '\r')
        if [ "$vulkan_ok" != "yes" ]; then
            echo ""
            echo "ERROR: Vulkan llama-server binary not found at ${DEVICE_BIN_VULKAN}"
            echo "  Build (Vulkan for Mali-G68):"
            echo "    cmake -B build-android-vulkan \\"
            echo "        -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK/build/cmake/android.toolchain.cmake \\"
            echo "        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \\"
            echo "        -DGGML_VULKAN=ON -DGGML_OPENMP=OFF"
            echo "    cmake --build build-android-vulkan --target llama-server -j\$(nproc)"
            echo "    adb push build-android-vulkan/bin/llama-server ${DEVICE_BIN_VULKAN}"
            echo "    adb shell chmod +x ${DEVICE_BIN_VULKAN}"
            echo ""
            echo "  Note: Mali-G68 Vulkan driver quality varies by ColorOS version."
            echo "  If inference crashes or gives garbage output, GPU layers may be unsupported."
            exit 1
        fi
        log "Vulkan binary OK: ${DEVICE_BIN_VULKAN}"
    fi
fi

# Phase 1: Discover thermal zones
banner "Phase 1: Discover thermal zones"
THERMAL_ZONES=""
if [ "$DRY_RUN" = 0 ]; then
    log "Discovering thermal zones (HAL → sysfs fallback)..."
    THERMAL_ZONES=$(discover_thermal_zones)
    if [ -z "$THERMAL_ZONES" ]; then
        log "  [WARN] No thermal zones found — thermal data will be empty"
    else
        log "  Found zones:"
        while IFS= read -r z; do log "    $z"; done <<< "$THERMAL_ZONES"
    fi
    echo "$THERMAL_ZONES" > "${BASE_ARTIFACT}/thermal_zones.txt"
fi

# Phase 2: Push model files to device
banner "Phase 2: Ensure models on device"

declare -a MODEL_NAMES MODEL_QUANTS MODEL_DEVICE_PATHS MODEL_TOKENIZERS MODEL_CONTEXT_SIZES

for entry in "${MODELS[@]}"; do
    IFS='|' read -r n q p t c <<< "$entry"
    MODEL_NAMES+=("$n")
    MODEL_QUANTS+=("$q")
    MODEL_DEVICE_PATHS+=("$p")
    MODEL_TOKENIZERS+=("$t")
    MODEL_CONTEXT_SIZES+=("$c")
done

if [ "$DRY_RUN" = 0 ] && [ -n "${RESUME_DIR:-}" ]; then
    log "Resume mode: skipping model downloads"
elif [ "$DRY_RUN" = 0 ]; then
    for i in "${!MODEL_NAMES[@]}"; do
        [[ -n "$ONLY_MODEL" && "${MODEL_NAMES[$i],,}" != *"${ONLY_MODEL,,}"* ]] && continue
        ensure_model_on_device \
            "${MODEL_NAMES[$i]}" \
            "${MODEL_DEVICE_PATHS[$i]}" || true
    done
fi

# Phase 3: Pollers are per-combo — nothing to start globally
banner "Phase 3: Start pollers"
if [ "$DRY_RUN" = 1 ]; then
    log "  [DRY RUN] Skipping pollers"
else
    log "  Pollers are per-combo — will start/stop around each aiperf run"
fi

# Phase 4: Model loop — one pass per backend
banner "Phase 4: Model loop  [backend=${BACKEND}]"

# Determine which backend passes to run
if [ "$BACKEND" = "both" ]; then
    BACKEND_PASSES=("cpu" "vulkan")
else
    BACKEND_PASSES=("$BACKEND")
fi

declare -a SKIPPED_MODELS SMOKE_PASSED_MODELS BENCH_COMPLETED

for CURRENT_BACKEND in "${BACKEND_PASSES[@]}"; do

if [ "${#BACKEND_PASSES[@]}" -gt 1 ]; then
    banner "Backend pass: ${CURRENT_BACKEND}"
fi

# Per-backend artifact subdir so cpu/vulkan results sit side by side
BACKEND_ARTIFACT="${BASE_ARTIFACT}/${CURRENT_BACKEND}"
mkdir -p "$BACKEND_ARTIFACT"

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_QUANT="${MODEL_QUANTS[$i]}"
    DEVICE_MODEL_PATH="${MODEL_DEVICE_PATHS[$i]}"
    MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
    CONTEXT_SIZE="${MODEL_CONTEXT_SIZES[$i]}"

    [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue

    if [ "$DRY_RUN" = 1 ]; then
        log "[DRY RUN] Would benchmark: ${MODEL_NAME} (${MODEL_QUANT}) [${CURRENT_BACKEND}]"
        BENCH_COMPLETED+=("${MODEL_NAME}:${CURRENT_BACKEND}")
        continue
    fi

    # Resume: count missing combos in the backend-specific artifact subdir
    MISSING_COMBOS=0
    for G in "${GEN_LENGTHS[@]}"; do
        for P in "${PROMPT_LENGTHS[@]}"; do
            [ ! -f "${BACKEND_ARTIFACT}/${MODEL_NAME}/gen${G}/ctx${P}/profile_export_aiperf.json" ] && \
                MISSING_COMBOS=$((MISSING_COMBOS + 1))
        done
    done

    if [ "$MISSING_COMBOS" = 0 ]; then
        log "  [RESUME SKIP] ${MODEL_NAME} [${CURRENT_BACKEND}] — all combos complete"
        BENCH_COMPLETED+=("${MODEL_NAME}:${CURRENT_BACKEND}")
        continue
    fi
    [ -n "$RESUME_DIR" ] && log "  [RESUME] ${MODEL_NAME} [${CURRENT_BACKEND}] — ${MISSING_COMBOS} combo(s) remaining"

    echo ""
    echo "  ┌──────────────────────────────────────────────────────┐"
    printf "  │  Model    : %-42s│\n" "${MODEL_NAME} (${MODEL_QUANT})"
    printf "  │  Backend  : %-42s│\n" "${CURRENT_BACKEND}"
    printf "  │  Tokenizer: %-42s│\n" "${MODEL_TOKENIZER}"
    printf "  │  Device   : %-42s│\n" "${DEVICE_MODEL_PATH}"
    echo "  └──────────────────────────────────────────────────────┘"

    # Verify model on device
    model_ok=$(adb shell "[ -f '${DEVICE_MODEL_PATH}' ] && echo yes || echo no" 2>/dev/null | tr -d '\r')
    if [ "$model_ok" != "yes" ]; then
        log "  [SKIP] Model not on device: ${DEVICE_MODEL_PATH}"
        SKIPPED_MODELS+=("${MODEL_NAME}:${CURRENT_BACKEND} (model not found on device)")
        continue
    fi

    # Setup port forward and start server with the current backend
    setup_port_forward

    if ! start_server "$DEVICE_MODEL_PATH" "$CONTEXT_SIZE" "$MODEL_NAME" "$CURRENT_BACKEND"; then
        log "  [FAIL] Could not start server for ${MODEL_NAME} [${CURRENT_BACKEND}]"
        SKIPPED_MODELS+=("${MODEL_NAME}:${CURRENT_BACKEND} (server failed to start)")
        kill_server
        continue
    fi

    # Smoke test
    if [ "$SKIP_SMOKE" = 1 ]; then
        log "  [SMOKE SKIP] --skip-smoke set"
        SMOKE_PASSED_MODELS+=("${MODEL_NAME}:${CURRENT_BACKEND} (smoke skipped)")
    else
        log "  ── Smoke [${CURRENT_BACKEND}] ─────────────────────────────────"
        SMOKE_PASS=1
        for tok_count in 32 128 256; do
            msg=$(python3 -c "print('The quick brown fox jumped over the lazy dog. ' * $((tok_count / 8 + 1)))")
            log "  Sending ~${tok_count}-tok prompt..."
            set +e
            resp=$(curl -s "${SERVER_URL}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${msg}\"}],\"max_tokens\":32}" \
                --max-time 60 2>/dev/null)
            CURL_EXIT=$?
            set -e

            if [ "$CURL_EXIT" != 0 ]; then
                log "  [!] curl failed (exit $CURL_EXIT) at ~${tok_count} tok"
                SMOKE_PASS=0; break
            fi
            set +e
            echo "$resp" | python3 -c \
                "import sys,json; d=json.load(sys.stdin); assert d.get('choices',[{}])[0].get('message')" \
                2>/dev/null
            PY_EXIT=$?
            set -e
            if [ "$PY_EXIT" = 0 ]; then
                log "  [OK] ~${tok_count} tok — valid response"
            else
                log "  [!] ~${tok_count} tok — bad response: ${resp:0:120}"
                SMOKE_PASS=0; break
            fi
        done
        log "  ──────────────────────────────────────────────────────"

        if [ "$SMOKE_PASS" = 0 ]; then
            log "  [SMOKE FAIL] ${MODEL_NAME} [${CURRENT_BACKEND}] — skipping benchmark"
            SKIPPED_MODELS+=("${MODEL_NAME}:${CURRENT_BACKEND} (smoke test failed)")
            kill_server; continue
        fi
        log "  [SMOKE PASS] ${MODEL_NAME} [${CURRENT_BACKEND}] — starting aiperf sweep"
        SMOKE_PASSED_MODELS+=("${MODEL_NAME}:${CURRENT_BACKEND}")
    fi

    echo "MODEL_START:${MODEL_NAME}:${CURRENT_BACKEND}:$(date +%s)" >> "$TIMING_LOG"

    # aiperf sweep — artifacts go into backend-specific subdir
    TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))
    RUN_NUM=0

    for GEN in "${GEN_LENGTHS[@]}"; do
        for CTX in "${PROMPT_LENGTHS[@]}"; do
            RUN_NUM=$((RUN_NUM + 1))
            ARTIFACT_DIR="${BACKEND_ARTIFACT}/${MODEL_NAME}/gen${GEN}/ctx${CTX}"
            mkdir -p "$ARTIFACT_DIR"

            if [ -f "${ARTIFACT_DIR}/profile_export_aiperf.json" ]; then
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  [RESUME SKIP]"
                continue
            fi

            log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS  [${CURRENT_BACKEND}]"

            if ! ensure_server_alive "$DEVICE_MODEL_PATH" "$CONTEXT_SIZE" "$MODEL_NAME" "$CURRENT_BACKEND"; then
                log "  [ABORT] Cannot recover server for ${MODEL_NAME} [${CURRENT_BACKEND}]"
                SKIPPED_MODELS+=("${MODEL_NAME}:${CURRENT_BACKEND} (server unrecoverable at gen=${GEN} ctx=${CTX})")
                break 2
            fi

            # Per-combo pollers — write directly into this combo's artifact dir
            THERMAL_LOG="${ARTIFACT_DIR}/thermal.csv"
            CPUFREQ_LOG="${ARTIFACT_DIR}/cpufreq.csv"
            BATTERY_LOG="${ARTIFACT_DIR}/battery.csv"

            # Combo metadata for report.py (quant, backend, gen, ctx without parsing timing_log)
            printf '{"model":"%s","quant":"%s","backend":"%s","gen":%d,"ctx":%d}\n' \
                "$MODEL_NAME" "$MODEL_QUANT" "$CURRENT_BACKEND" "$GEN" "$CTX" \
                > "${ARTIFACT_DIR}/combo_info.json"

            stop_thermal_poller; stop_cpufreq_poller; stop_battery_poller
            [ -n "$THERMAL_ZONES" ] && start_thermal_poller "$THERMAL_ZONES"
            start_cpufreq_poller
            start_battery_poller

            "$AIPERF_BIN" profile \
                --model                         "${MODEL_NAME}" \
                --streaming \
                --endpoint-type                 'chat' \
                --url                           "${SERVER_URL}" \
                --tokenizer                     "${MODEL_TOKENIZER}" \
                --synthetic-input-tokens-mean   "${CTX}" \
                --synthetic-input-tokens-stddev 0 \
                --output-tokens-mean            "${GEN}" \
                --request-count                 "${REQS}" \
                --concurrency                   "${CONCURRENCY}" \
                --slice-duration                "${SLICE_DURATION}" \
                --random-seed                   "${RANDOM_SEED}" \
                --request-timeout-seconds       "${REQUEST_TIMEOUT}" \
                --artifact-dir                  "${ARTIFACT_DIR}" \
                || log "  aiperf failed (ctx=${CTX} gen=${GEN})"

            stop_thermal_poller; stop_cpufreq_poller; stop_battery_poller

            if [ "$RUN_NUM" -lt "$TOTAL_RUNS" ]; then
                log "  Cooldown ${COOLDOWN_COMBO}s..."
                sleep "$COOLDOWN_COMBO"
            fi
        done
    done

    echo "MODEL_END:${MODEL_NAME}:${CURRENT_BACKEND}:$(date +%s)" >> "$TIMING_LOG"
    BENCH_COMPLETED+=("${MODEL_NAME}:${CURRENT_BACKEND}")

    # Battery level snapshot at model end
    batt_end=$(adb shell "dumpsys battery 2>/dev/null | awk '/level:/{print \$2}'" | tr -d '\r')
    batt_drain=$(( ${batt_start:-0} - ${batt_end:-0} ))
    log "  Battery at model end: ${batt_end}%  (drained: ${batt_drain}%)"
    echo "BATTERY_SNAP:${MODEL_NAME}:${CURRENT_BACKEND}:end:${batt_end}:$(date +%s%3N)" >> "$BATTERY_LOG"

    kill_server
    log "  Model done. Thermal cooldown ${COOLDOWN_MODEL}s..."
    sleep "$COOLDOWN_MODEL"
done  # model loop

# Inter-backend cooldown when running 'both'
if [ "${#BACKEND_PASSES[@]}" -gt 1 ] && [ "$CURRENT_BACKEND" = "cpu" ]; then
    log "Backend switch: cpu → vulkan. Cooldown ${COOLDOWN_BACKEND}s for thermals to settle..."
    sleep "$COOLDOWN_BACKEND"
fi

done  # backend loop

# Phase 5: Pollers already stopped per-combo — just ensure clean state
banner "Phase 5: Stop pollers"
if [ "$DRY_RUN" = 0 ]; then
    stop_thermal_poller; stop_cpufreq_poller; stop_battery_poller
    log "Final battery level:"
    adb shell dumpsys battery 2>/dev/null | tr -d '\r' \
        | grep -E "(level|temperature|voltage):" | sed 's/^/  /'
    log "Restoring device state..."
    adb shell svc data enable 2>/dev/null || true
    adb shell cmd power set-fixed-performance-mode-enabled false 2>/dev/null || true
    adb shell svc power stayon false 2>/dev/null || true
    adb shell input keyevent 26 2>/dev/null || true
fi

if [ "$DRY_RUN" = 1 ]; then
    banner "Dry run complete"; exit 0
fi

# Phase 6: Summary
banner "Phase 6: Summary"
echo ""
echo "  COMPLETED (${#BENCH_COMPLETED[@]})"
for m in "${BENCH_COMPLETED[@]}"; do echo "    [OK]   $m"; done
echo ""
echo "  SMOKE PASSED (${#SMOKE_PASSED_MODELS[@]})"
for m in "${SMOKE_PASSED_MODELS[@]}"; do echo "    [OK]   $m"; done
echo ""
echo "  SKIPPED / FAILED (${#SKIPPED_MODELS[@]})"
for m in "${SKIPPED_MODELS[@]}"; do echo "    [FAIL] $m"; done
echo ""

if [ "${#BENCH_COMPLETED[@]}" = 0 ]; then
    echo "  No models completed — no report generated."; exit 1
fi

# Phase 7: Generate report
banner "Phase 7: Generating report"
REPORT_PY_PATH="${BASE_ARTIFACT}/generate_report.py"
cp "${SCRIPT_DIR}/generate_report.py" "$REPORT_PY_PATH" 2>/dev/null || true

if [ -f "$REPORT_PY_PATH" ]; then
    SKIPPED_STR=""
    for s in "${SKIPPED_MODELS[@]}"; do SKIPPED_STR="${SKIPPED_STR}${s}||"; done

    python3 "$REPORT_PY_PATH" \
        "$BASE_ARTIFACT" \
        "$THERMAL_LOG" \
        "$TIMING_LOG" \
        "${SKIPPED_STR}" \
        "${CONTEXT_SIZE}" \
        "${BACKEND}" \
        "$(device_model 2>/dev/null || echo unknown)" \
    && log "Report generated: ${BASE_ARTIFACT}/report.md" \
    || log "[WARN] Report generation failed — raw data still in ${BASE_ARTIFACT}"
else
    log "[WARN] generate_report.py not found — run manually after copying from Jetson benchmark dir"
fi

banner "Done"
echo "  Backend    : ${BACKEND}"
echo "  Device     : $(cat "$DEVICE_INFO_FILE" 2>/dev/null | grep 'Model:' | cut -d: -f2 | xargs)"
echo "  Artifacts  : ${BASE_ARTIFACT}"
echo "  Structure  : ${BASE_ARTIFACT}/cpu/   and   ${BASE_ARTIFACT}/vulkan/"
echo "  Thermal    : ${THERMAL_LOG}"
echo ""
echo "  Next steps:"
echo "    python3 generate_report.py ${BASE_ARTIFACT}"
echo "    aiperf plot ${BASE_ARTIFACT}/cpu --dashboard --port 8050"
echo "    aiperf plot ${BASE_ARTIFACT}/vulkan --dashboard --port 8051"
echo ""