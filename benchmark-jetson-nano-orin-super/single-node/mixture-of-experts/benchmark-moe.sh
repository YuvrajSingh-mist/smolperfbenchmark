#!/bin/bash
# benchmark-moe.sh — single-node llama.cpp benchmark for small MoE models that
# fit entirely on ONE Jetson Orin Nano Super 8GB (no RPC / no second node).
#
# Per model: (re)build llama.cpp locally (CUDA only, no RPC) → ONE server launch
# → smoke → aiperf sweep. Sweeps: prompt in {128,512,1024,2048} × gen in
# {64,128,256}. Key metric: output tok/J (output tokens per joule), from
# aiperf + tegrastats.
#
# Usage:
#   bash benchmark-moe.sh --power-mode 1          # 25W — recommended
#   bash benchmark-moe.sh --power-mode 1 --only lfm2
#   bash benchmark-moe.sh --reqs 5
#   bash benchmark-moe.sh --skip-smoke
#   bash benchmark-moe.sh --skip-build
#   bash benchmark-moe.sh --dry-run
#   bash benchmark-moe.sh --resume DIR --power-mode 1
#
# NOTE: llama.cpp is (re)built on THIS node with -DGGML_CUDA=ON only (no
# -DGGML_RPC=ON — that flag/binary was only needed for the multi-node cluster
# variant, see ../../multi-node/mixture-of-experts/). Pass --skip-build to reuse
# an existing $LLAMACPP_BIN instead of rebuilding. Models are downloaded
# just-in-time and deleted after their sweep. Without --power-mode, the default
# 4-mode power sweep will re-download every model once per mode (~4x bandwidth)
# — pass --power-mode explicitly to avoid that.

set -euo pipefail

# Pip installs user-local binaries here; add unconditionally so all sub-shells see it
export PATH="$HOME/.local/bin:$PATH"

# ── Ensure tmux is installed ──────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
    echo "tmux not found — installing..."
    sudo apt-get update -qq && sudo apt-get install -y tmux
fi

# ── Ensure Hugging Face CLI is installed ──────────────────────────────────────
if ! command -v hf &>/dev/null; then
    echo "Hugging Face CLI (hf) not found — installing..."

    if command -v apt-get &>/dev/null; then
        echo "Attempting system package install..."
        sudo apt-get update -qq && sudo apt-get install -y python3-huggingface-hub 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v hf &>/dev/null; then
        echo "System package not available, using pip..."
        if command -v pip3 &>/dev/null; then
            PIP_CMD="pip3"
        elif command -v pip &>/dev/null; then
            PIP_CMD="pip"
        else
            echo "ERROR: Neither pip3 nor pip found. Install with:"
            echo "  sudo apt-get install -y python3-pip"
            exit 1
        fi
        $PIP_CMD install --break-system-packages -U "huggingface_hub[cli]" || \
        $PIP_CMD install --user -U "huggingface_hub[cli]"
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v hf &>/dev/null; then
        echo "ERROR: hf command still not available after install attempts."
        echo "Try manually: pip3 install --break-system-packages 'huggingface_hub[cli]'"
        exit 1
    fi
    echo "Hugging Face CLI installed successfully."
fi

# ── Auto-relaunch inside tmux if not already there ────────────────────────────
if [ -z "${TMUX:-}" ]; then
    SESSION="moe-bench"
    RELAUNCH_ARGS=""
    [ $# -gt 0 ] && RELAUNCH_ARGS=$(printf '%q ' "$@")
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "bash $(realpath "$0") $RELAUNCH_ARGS; echo 'Done — press Enter to exit'; read"
    echo "Launched in tmux session '$SESSION'"
    echo "Attach with:  tmux attach -t $SESSION"
    exit 0
fi

# ── Config ────────────────────────────────────────────────────────────────────
REQS=20
ONLY_MODEL=""
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
POWER_MODE=0
POWER_MODE_NAME="15w"
POWER_MODE_EXPLICIT=0
CONCURRENCY=1
SLICE_DURATION=30
RANDOM_SEED=42
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10
COOLDOWN_MODEL=30
SERVER_STARTUP_TIMEOUT=300

PROMPT_LENGTHS=(128 512 1024 2048)
GEN_LENGTHS=(64 128 256)
CONTEXT_SIZE=2560   # max_prompt(2048) + max_gen(256) = 2304, padded to 2560

LLAMACPP_BIN="${LLAMACPP_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
LLAMACPP_PORT=8080

# ── Local build config (single node — no RPC) ─────────────────────────────────
LLAMACPP_SRC_DIR="$HOME/llama.cpp"
CUDA_ARCH="${CUDA_ARCH:-87}"   # Jetson Orin (Ampere, SM 8.7)
REBUILD_LLAMACPP=1             # rebuild fresh (no -DGGML_RPC=ON) unless --skip-build
GGUF_DIR="$HOME/moe-gguf-models"   # JIT scratch dir, cleaned per combo

TEGRA_PIDFILE="/tmp/moe_bench_tegrastats.pid"
COMBO_TEGRA_PIDFILE="/tmp/moe_bench_combo_tegrastats.pid"
SERVER_PIDFILE="/tmp/moe_bench_server.pid"
REPORT_PY="/tmp/moe_report.py"

# ── Single-node MoE model table: name|quant|hf_repo|hf_file|tokenizer|ctx_size ─
# Small MoE models whose full (total, not just active) parameter count fits
# resident in 8GB unified memory at Q4_K_M, alongside context/KV cache and the
# OS. Downloaded just-in-time and deleted after their sweep. For MoE models
# too big to fit a single 8GB node (gpt-oss-20b, Qwen3-30B-A3B,
# granite-4.0-h-small, ...) see ../../multi-node/mixture-of-experts/.
declare -a MOE_MODELS=(
    "smallthinker-4b-a0.6b|Q4_K_M|gabriellarson/SmallThinker-4BA0.6B-Instruct-GGUF|SmallThinker-4BA0.6B-Instruct-Q4_K_M.gguf|PowerInfer/SmallThinker-4BA0.6B-Instruct|2560"
    "trinity-nano|Q4_K_M|arcee-ai/Trinity-Nano-Preview-GGUF|Trinity-Nano-Preview-Q4_K_M.gguf|arcee-ai/Trinity-Nano-Preview|2560"
    "granite4-h-tiny|Q4_K_M|unsloth/granite-4.0-h-tiny-GGUF|granite-4.0-h-tiny-Q4_K_M.gguf|ibm-granite/granite-4.0-h-tiny|2560"
    "lfm2.5-8b-a1b|Q4_K_M|LiquidAI/LFM2.5-8B-A1B-GGUF|LFM2.5-8B-A1B-Q4_K_M.gguf|LiquidAI/LFM2.5-8B-A1B|2560"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reqs)        REQS="$2";         shift 2 ;;
        --only)        ONLY_MODEL="$2";   shift 2 ;;
        --skip-smoke)  SKIP_SMOKE=1;      shift ;;
        --skip-build)  REBUILD_LLAMACPP=0; shift ;;
        --dry-run)     DRY_RUN=1;         shift ;;
        --resume)      RESUME_DIR="$2";   shift 2 ;;
        --maxn)        POWER_MODE=2; POWER_MODE_NAME="maxn"; POWER_MODE_EXPLICIT=1; shift ;;
        --power-mode)  POWER_MODE="$2"; POWER_MODE_EXPLICIT=1
                       case "$POWER_MODE" in
                           0) POWER_MODE_NAME="15w"  ;;
                           1) POWER_MODE_NAME="25w"  ;;
                           2) POWER_MODE_NAME="maxn" ;;
                           3) POWER_MODE_NAME="7w"   ;;
                           *) POWER_MODE_NAME="pwr${POWER_MODE}" ;;
                       esac
                       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Sweep setup ───────────────────────────────────────────────────────────────
SWEEP_DATE=$(date +%Y%m%d-%H%M)
if [ -n "$RESUME_DIR" ] || [ "$POWER_MODE_EXPLICIT" = 1 ]; then
    SWEEP_MODES=("$POWER_MODE")
    SWEEP_NAMES=("$POWER_MODE_NAME")
else
    SWEEP_MODES=(2 1 0 3)
    SWEEP_NAMES=(maxn 25w 15w 7w)
    echo "  [WARN] No --power-mode given — defaulting to a 4-mode sweep. Each model"
    echo "         is deleted after its sweep, so this re-downloads all 3 models"
    echo "         once per mode (~4x, up to ~60GB total). Pass --power-mode to avoid this."
fi
# BASE_ARTIFACT and TIMING_LOG are resolved per-mode inside the sweep loop

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
log()    { echo "  [$(date +%H:%M:%S)] $*"; }

stop_tegrastats() {
    [ -f "$TEGRA_PIDFILE" ] && sudo kill "$(cat "$TEGRA_PIDFILE")" 2>/dev/null || true
    rm -f "$TEGRA_PIDFILE"
    [ -f "$COMBO_TEGRA_PIDFILE" ] && sudo kill "$(cat "$COMBO_TEGRA_PIDFILE")" 2>/dev/null || true
    rm -f "$COMBO_TEGRA_PIDFILE"
    sudo pkill -f "tegrastats" 2>/dev/null || true
}

# Per-combo tegrastats: each aiperf run gets its own tegrastats.log in its artifact dir.
# This makes resume safe — done combos keep their power data, new combos get fresh captures.
start_combo_tegrastats() {
    local artifact_dir="$1"
    [ -f "$COMBO_TEGRA_PIDFILE" ] && sudo kill "$(cat "$COMBO_TEGRA_PIDFILE")" 2>/dev/null || true
    rm -f "$COMBO_TEGRA_PIDFILE"
    [ "$DRY_RUN" = 1 ] && return
    sudo tegrastats --interval 500 --logfile "$artifact_dir/tegrastats.log" &
    echo $! > "$COMBO_TEGRA_PIDFILE"
}

stop_combo_tegrastats() {
    [ -f "$COMBO_TEGRA_PIDFILE" ] && sudo kill "$(cat "$COMBO_TEGRA_PIDFILE")" 2>/dev/null || true
    rm -f "$COMBO_TEGRA_PIDFILE"
}

# Snapshot RSS (VmRSS) of llama-server on this node. Appends to
# <artifact_dir>/rss.log. Cheap point-in-time read via /proc, called once
# before and once after each aiperf run rather than backgrounded.
sample_rss_snapshot() {
    local artifact_dir="$1" label="$2"
    [ "$DRY_RUN" = 1 ] && return
    {
        echo "=== $label $(date '+%Y-%m-%d %H:%M:%S') ==="
        if [ -f "$SERVER_PIDFILE" ] && kill -0 "$(cat "$SERVER_PIDFILE" 2>/dev/null)" 2>/dev/null; then
            local rss_kb
            rss_kb=$(awk '/VmRSS/{print $2}' "/proc/$(cat "$SERVER_PIDFILE")/status" 2>/dev/null || echo "?")
            printf "  %s: %s kB\n" "$(hostname)" "$rss_kb"
        fi
    } >> "$artifact_dir/rss.log"
}

kill_llamacpp() {
    if [ -f "$SERVER_PIDFILE" ]; then
        kill "$(cat "$SERVER_PIDFILE")" 2>/dev/null || true
        rm -f "$SERVER_PIDFILE"
    fi
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null || true
    # Jetson unified memory doesn't release instantly
    sleep 12
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
    sleep 2
}

ensure_llamacpp_alive() {
    local model_path="$1" ctx_size="$2" srv_log="$3"
    local code
    code=$(curl -s "http://localhost:$LLAMACPP_PORT/v1/models" --max-time 3 \
           -o /dev/null -w "%{http_code}" 2>/dev/null || true)
    [ "$code" = "200" ] && return 0

    log "  [!] llama-server not responding (HTTP $code) — restarting..."
    kill_llamacpp
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
    sleep 3
    "$LLAMACPP_BIN" -m "$model_path" --host 0.0.0.0 --port "$LLAMACPP_PORT" \
        -ngl 99 --parallel 1 -c "$ctx_size" \
        --no-cache-prompt --cache-ram 0 \
        >> "$srv_log" 2>&1 &
    echo $! > "$SERVER_PIDFILE"
    local elapsed=0
    while [ "$elapsed" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
        sleep 2; elapsed=$((elapsed + 2))
        ! kill -0 "$(cat "$SERVER_PIDFILE" 2>/dev/null)" 2>/dev/null && \
            log "  [RESTART FAIL] server died" && return 1
        code=$(curl -s "http://localhost:$LLAMACPP_PORT/v1/models" --max-time 3 \
               -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        [ "$code" = "200" ] && log "  [RESTART OK] t=${elapsed}s" && return 0
    done
    log "  [RESTART FAIL] timed out after ${SERVER_STARTUP_TIMEOUT}s"; return 1
}

# ── llama.cpp: local (re)build, no RPC ────────────────────────────────────────
# Small MoE archs (lfm2moe, granitehybrid, olmoe) landed in llama.cpp fairly
# recently, so a stale checkout may not recognize these models — always pull +
# rebuild by default. Pass --skip-build to reuse the existing $LLAMACPP_BIN.
build_llamacpp() {
    banner "Building llama.cpp (CUDA only, no RPC)"
    if [ ! -d "$LLAMACPP_SRC_DIR/.git" ]; then
        log "Cloning llama.cpp into $LLAMACPP_SRC_DIR..."
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMACPP_SRC_DIR"
    fi
    ( cd "$LLAMACPP_SRC_DIR" && git pull --ff-only )
    cmake -S "$LLAMACPP_SRC_DIR" -B "$LLAMACPP_SRC_DIR/build" \
        -DGGML_CUDA=ON -DGGML_RPC=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMACPP_SRC_DIR/build" --config Release -j"$(nproc)" --target llama-server
    [ -f "$LLAMACPP_BIN" ] || { log "[ABORT] build finished but $LLAMACPP_BIN not found"; exit 1; }
    log "Build OK: $LLAMACPP_BIN"
}

# ── Smoke test ─────────────────────────────────────────────────────────────────
smoke_test() {
    local url="$1" model_tag="$2"
    log "  ── Smoke ─────────────────────────────────────────────"
    local pass=1
    for tok_count in 32 256 512; do
        log "  Sending ~${tok_count}-tok prompt..."
        # Use Python's json.dumps to guarantee valid JSON — bash string interpolation
        # is not safe for JSON construction (special chars, newlines, quotes in content).
        local payload
        payload=$(python3 -c "
import json, sys
content = 'The quick brown fox jumped over the lazy dog. ' * $((tok_count / 8 + 1))
print(json.dumps({'model': sys.argv[1], 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 32}))
" "$model_tag")
        set +e
        local resp curl_exit
        resp=$(curl -s "$url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            --max-time 120 2>/dev/null)
        curl_exit=$?
        set -e
        if [ "$curl_exit" != 0 ]; then
            log "  [!] curl failed (exit $curl_exit) at ~${tok_count} tok"
            pass=0; break
        fi
        set +e
        echo "$resp" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); assert d.get('choices',[{}])[0].get('message')" \
            2>/dev/null
        local py_exit=$?
        set -e
        if [ "$py_exit" = 0 ]; then
            log "  [OK] ~${tok_count} tok"
        else
            log "  [!] ~${tok_count} tok — bad response: ${resp:0:120}"
            pass=0; break
        fi
    done
    log "  ──────────────────────────────────────────────────────"
    return $((1 - pass))
}

# ── Parse MoE model table ─────────────────────────────────────────────────────
declare -a MOE_MODEL_NAMES MOE_MODEL_QUANTS MOE_HF_REPOS MOE_HF_FILES MOE_MODEL_TOKENIZERS MOE_MODEL_CTX_SIZES

for entry in "${MOE_MODELS[@]}"; do
    IFS='|' read -r n q repo file t c <<< "$entry"
    MOE_MODEL_NAMES+=("$n")
    MOE_MODEL_QUANTS+=("$q")
    MOE_HF_REPOS+=("$repo")
    MOE_HF_FILES+=("$file")
    MOE_MODEL_TOKENIZERS+=("$t")
    MOE_MODEL_CTX_SIZES+=("${c:-$CONTEXT_SIZE}")
done

# ── Print config banner ───────────────────────────────────────────────────────
banner "MoE Single-Node Benchmark  |  1× Jetson Orin Nano Super 8GB"
echo "  Date      : $(date --iso-8601=seconds)"
echo "  Node      : $(hostname)"
echo "  Requests  : $REQS per run"
echo "  Prompts   : ${PROMPT_LENGTHS[*]}"
echo "  Gen lens  : ${GEN_LENGTHS[*]}"
if [ "${#SWEEP_MODES[@]}" -gt 1 ]; then
    echo "  Power     : sweep ${SWEEP_NAMES[*]}"
else
    echo "  Power     : mode ${SWEEP_MODES[0]} (${SWEEP_NAMES[0]})"
fi
[ -n "$ONLY_MODEL" ] && echo "  Filter    : $ONLY_MODEL"
[ "$DRY_RUN"   = 1 ] && echo "  Mode      : DRY RUN"
[ -n "$RESUME_DIR" ] && echo "  Mode      : RESUME ($RESUME_DIR)"

# ── Initial cleanup (once) ────────────────────────────────────────────────────
banner "Cleanup"
pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null && log "killed llama-server" || true
sudo pkill -f "tegrastats" 2>/dev/null && log "killed tegrastats" || true
rm -f "$TEGRA_PIDFILE" "$SERVER_PIDFILE"
sleep 2
echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
CMA_FREE_KB=$(awk '/CmaFree/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
log "CMA free: $(( CMA_FREE_KB / 1024 )) MiB"

# ── Build llama.cpp (once, no RPC) ────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ] && [ "$REBUILD_LLAMACPP" = 1 ]; then
    build_llamacpp
else
    [ -f "$LLAMACPP_BIN" ] || { echo "ERROR: llama-server not found: $LLAMACPP_BIN (drop --skip-build to build it)"; exit 1; }
    log "Skipping build — reusing $LLAMACPP_BIN"
fi

# ── Activate aiperf venv (once) ──────────────────────────────────────────────
source "$HOME/venv/bin/activate" 2>/dev/null || \
source "$HOME/aiperf-env/bin/activate" 2>/dev/null || \
{ echo "ERROR: no aiperf venv found (~venv or ~aiperf-env)"; exit 1; }

declare -a SKIPPED_MODELS BENCH_MODELS

# ══════════════════════════════════════════════════════════════════════════════
# Single-node run (this Jetson only — no RPC)
# ══════════════════════════════════════════════════════════════════════════════
run_moe_single_node() {
    banner "MoE single-node: $(hostname)"
    [ -f "$LLAMACPP_BIN" ] || { log "[ABORT] llama-server not found: $LLAMACPP_BIN"; return; }

    mkdir -p "$GGUF_DIR"

    for i in "${!MOE_MODEL_NAMES[@]}"; do
        local MODEL_NAME="${MOE_MODEL_NAMES[$i]}"
        local MODEL_QUANT="${MOE_MODEL_QUANTS[$i]}"
        local HF_REPO="${MOE_HF_REPOS[$i]}"
        local HF_FILE="${MOE_HF_FILES[$i]}"
        local MODEL_TOKENIZER="${MOE_MODEL_TOKENIZERS[$i]}"
        local MODEL_CTX_SIZE="${MOE_MODEL_CTX_SIZES[$i]}"
        local COMBO_NAME="${MODEL_NAME}-${MODEL_QUANT}"
        local MODEL_PATH="$GGUF_DIR/${COMBO_NAME}.gguf"

        [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue
        [ "$DRY_RUN" = 1 ] && { log "[DRY RUN] $COMBO_NAME  ($HF_REPO/$HF_FILE)"; continue; }

        # Resume: skip if all combos already done
        local missing=0
        for G in "${GEN_LENGTHS[@]}"; do
            for P in "${PROMPT_LENGTHS[@]}"; do
                [ ! -f "$BASE_ARTIFACT/llamacpp/$COMBO_NAME/gen${G}/ctx${P}/profile_export_aiperf.json" ] && \
                    missing=$((missing + 1))
            done
        done
        if [ "$missing" = 0 ]; then
            log "  [RESUME SKIP] $COMBO_NAME — all combos done"
            BENCH_MODELS+=("llamacpp:$i")
            continue
        fi

        echo ""
        echo "  ┌─────────────────────────────────────────────────────┐"
        printf "  │  %-52s│\n" "$COMBO_NAME"
        echo "  └─────────────────────────────────────────────────────┘"

        log "  Downloading $HF_FILE from $HF_REPO (JIT — deleted after this combo)..."
        if ! hf download "$HF_REPO" "$HF_FILE" --local-dir "$GGUF_DIR"; then
            log "  [SKIP] download failed: $HF_REPO/$HF_FILE"
            SKIPPED_MODELS+=("$COMBO_NAME (download failed)")
            continue
        fi
        mv "$GGUF_DIR/$HF_FILE" "$MODEL_PATH"

        local SERVER_LOG="$BASE_ARTIFACT/llamacpp/${COMBO_NAME}-server.log"
        mkdir -p "$(dirname "$SERVER_LOG")"
        log "Launching llama-server on :$LLAMACPP_PORT ..."
        "$LLAMACPP_BIN" -m "$MODEL_PATH" \
            --host 0.0.0.0 --port "$LLAMACPP_PORT" \
            -ngl 99 --parallel 1 -c "$MODEL_CTX_SIZE" \
            --no-cache-prompt --cache-ram 0 \
            > "$SERVER_LOG" 2>&1 &
        echo $! > "$SERVER_PIDFILE"
        log "  PID: $(cat "$SERVER_PIDFILE")"

        local READY=0 ELAPSED=0
        while [ "$ELAPSED" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
            sleep 3; ELAPSED=$((ELAPSED + 3))
            if ! kill -0 "$(cat "$SERVER_PIDFILE" 2>/dev/null)" 2>/dev/null; then
                log "  [!] Server died at t=${ELAPSED}s (OOM?)"
                grep -E "error|OOM|failed|CUDA" "$SERVER_LOG" | tail -8 | sed 's/^/      /'
                kill_llamacpp; break
            fi
            local CODE
            CODE=$(curl -s "http://localhost:$LLAMACPP_PORT/v1/models" --max-time 3 \
                   -o /dev/null -w "%{http_code}" 2>/dev/null || true)
            if   [ "$CODE" = "200" ]; then READY=1; log "  [OK] HTTP 200 at t=${ELAPSED}s"; break
            elif [ "$CODE" = "503" ]; then log "  t=${ELAPSED}s: loading weights..."
            else                           log "  t=${ELAPSED}s: HTTP $CODE"; fi
        done

        if [ "$READY" = 0 ]; then
            log "  [FAIL] $COMBO_NAME — server did not start"
            SKIPPED_MODELS+=("$COMBO_NAME (server failed)")
            kill_llamacpp; rm -f "$MODEL_PATH"; continue
        fi

        if [ "$SKIP_SMOKE" = 1 ]; then
            log "  [SMOKE SKIP]"
        else
            if ! smoke_test "http://localhost:$LLAMACPP_PORT" "$MODEL_NAME"; then
                log "  [SMOKE FAIL] $COMBO_NAME — skipping"
                SKIPPED_MODELS+=("$COMBO_NAME (smoke failed)")
                kill_llamacpp; rm -f "$MODEL_PATH"; continue
            fi
            log "  [SMOKE PASS]"
        fi

        BENCH_MODELS+=("llamacpp:$i")

        local RUN_NUM=0
        local TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))
        for GEN in "${GEN_LENGTHS[@]}"; do
            for CTX in "${PROMPT_LENGTHS[@]}"; do
                RUN_NUM=$((RUN_NUM + 1))
                local ARTIFACT_DIR="$BASE_ARTIFACT/llamacpp/$COMBO_NAME/gen${GEN}/ctx${CTX}"
                mkdir -p "$ARTIFACT_DIR"
                [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [RESUME SKIP]"; continue; }
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"
                if ! ensure_llamacpp_alive "$MODEL_PATH" "$MODEL_CTX_SIZE" "$SERVER_LOG"; then
                    log "  [ABORT] Cannot recover server"
                    SKIPPED_MODELS+=("$COMBO_NAME (server unrecoverable at gen=$GEN ctx=$CTX)")
                    stop_combo_tegrastats
                    break 2
                fi
                printf '{"model":"%s","quant":"%s","backend":"llamacpp","gen":%d,"ctx":%d}\n' \
                    "$MODEL_NAME" "$MODEL_QUANT" "$GEN" "$CTX" > "$ARTIFACT_DIR/combo_info.json"
                start_combo_tegrastats "$ARTIFACT_DIR"
                sample_rss_snapshot "$ARTIFACT_DIR" "pre-aiperf"
                aiperf profile \
                    --model                         "$MODEL_NAME" \
                    --streaming \
                    --endpoint-type                 'chat' \
                    --url                           "http://localhost:$LLAMACPP_PORT" \
                    --tokenizer                     "$MODEL_TOKENIZER" \
                    --synthetic-input-tokens-mean   "$CTX" \
                    --synthetic-input-tokens-stddev 0 \
                    --output-tokens-mean            "$GEN" \
                    --request-count                 "$REQS" \
                    --concurrency                   "$CONCURRENCY" \
                    --slice-duration                "$SLICE_DURATION" \
                    --random-seed                   "$RANDOM_SEED" \
                    --request-timeout-seconds       "$REQUEST_TIMEOUT" \
                    --artifact-dir                  "$ARTIFACT_DIR" \
                    || log "  aiperf failed (ctx=$CTX gen=$GEN)"
                sample_rss_snapshot "$ARTIFACT_DIR" "post-aiperf"
                stop_combo_tegrastats
                [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && { log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"; }
            done
        done

        kill_llamacpp
        rm -f "$MODEL_PATH"
        log "  [CLEANUP] Deleted $MODEL_PATH"
        log "Cooling ${COOLDOWN_MODEL}s..."
        sleep "$COOLDOWN_MODEL"
    done
}

# ══════════════════════════════════════════════════════════════════════════════
# POWER MODE SWEEP: MAXN → 25W → 15W → 7W
# ══════════════════════════════════════════════════════════════════════════════
for _sweep_idx in "${!SWEEP_MODES[@]}"; do
    POWER_MODE="${SWEEP_MODES[$_sweep_idx]}"
    POWER_MODE_NAME="${SWEEP_NAMES[$_sweep_idx]}"

    if [ -n "$RESUME_DIR" ]; then
        BASE_ARTIFACT="$RESUME_DIR"
    else
        BASE_ARTIFACT="$HOME/Desktop/benchmark/smolbenchmark/benchmark-jetson-nano-orin-super/single-node/mixture-of-experts/artifacts/blog-all-${SWEEP_DATE}-${POWER_MODE_NAME}"
    fi
    TIMING_LOG="$BASE_ARTIFACT/model_timing.log"
    mkdir -p "$BASE_ARTIFACT"

    SKIPPED_MODELS=()
    BENCH_MODELS=()

    banner "Power mode $(( _sweep_idx + 1 ))/${#SWEEP_MODES[@]}: $POWER_MODE_NAME  →  $BASE_ARTIFACT"

    # ── Power mode + clock lock ───────────────────────────────────────────────
    if [ "$DRY_RUN" = 0 ]; then
        log "Setting nvpmodel -m $POWER_MODE ($POWER_MODE_NAME)..."
        current_mode_id=$(sudo nvpmodel -q 2>/dev/null | awk 'NR==2{print $1}' || echo "-1")
        if [ "$current_mode_id" = "$POWER_MODE" ]; then
            log "Already at mode $POWER_MODE ($POWER_MODE_NAME) — skipping nvpmodel"
        else
            nvp_out=$(echo "yes" | sudo nvpmodel -m "$POWER_MODE" 2>&1) && true || true
            log "nvpmodel: $(echo "$nvp_out" | tr '\n' ' ')"
            sleep 1
        fi
        active_mode=$(sudo nvpmodel -q 2>/dev/null | head -1 || echo "unknown")
        log "Active power mode: $active_mode"
        if ! echo "$active_mode" | grep -qi "${POWER_MODE_NAME}"; then
            log "ERROR: active mode ($active_mode) does not match requested $POWER_MODE_NAME."
            log "  The device may still need a reboot. Run:"
            log "    echo yes | sudo nvpmodel -m $POWER_MODE && sudo reboot"
            log "  Then re-run with:  bash benchmark-moe.sh --power-mode $POWER_MODE"
            stop_combo_tegrastats; exit 1
        fi
        log "Power mode verified OK: $active_mode"
        sudo jetson_clocks       2>/dev/null && log "jetson_clocks OK"       || log "jetson_clocks not available"
        sudo jetson_clocks --fan 2>/dev/null && log "jetson_clocks --fan OK" || log "fan control not available"
    fi

    # ── Run single-node benchmark ─────────────────────────────────────────────
    banner "Running MoE single-node benchmarks (power=$POWER_MODE_NAME)"
    run_moe_single_node

    stop_combo_tegrastats  # safety: ensure no combo tegrastats left running

    if [ "$DRY_RUN" = 1 ]; then
        banner "Dry run complete ($POWER_MODE_NAME)"; continue
    fi
    if [ "${#BENCH_MODELS[@]}" = 0 ]; then
        echo "  No models benchmarked for $POWER_MODE_NAME."; continue
    fi

    # ── Generate report.md ────────────────────────────────────────────────────
    banner "Generating report.md ($POWER_MODE_NAME)"

cat > "$REPORT_PY" << 'PYEOF'
import re, sys, json, os, glob
from datetime import datetime

base_dir    = sys.argv[1]
skipped_arg = sys.argv[2] if len(sys.argv) > 2 else ""
report_path = os.path.join(base_dir, "report.md")

skipped = [s for s in skipped_arg.split("||") if s] if skipped_arg else []

# ── Per-combo power: each combo has its own tegrastats.log in its artifact dir ─
# Average all samples — no time-window slicing needed.
def parse_combo_power(artifact_dir):
    tlog = os.path.join(artifact_dir, "tegrastats.log")
    pw, cpu, gpu, tj = [], [], [], []
    try:
        for line in open(tlog):
            pw_m  = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
            cpu_m = re.search(r'cpu@([\d.]+)C', line)
            gpu_m = re.search(r'gpu@([\d.]+)C', line)
            tj_m  = re.search(r'tj@([\d.]+)C',  line)
            if pw_m:
                pw.append(int(pw_m.group(1)) / 1000.0)
                if cpu_m: cpu.append(float(cpu_m.group(1)))
                if gpu_m: gpu.append(float(gpu_m.group(1)))
                if tj_m:  tj.append(float(tj_m.group(1)))
    except FileNotFoundError:
        return None, None, None, None
    if not pw: return None, None, None, None
    return (
        sum(pw)  / len(pw),
        sum(cpu) / len(cpu) if cpu else None,
        sum(gpu) / len(gpu) if gpu else None,
        max(tj)             if tj  else None,
    )

def get_combo_quant(artifact_dir):
    try:
        return json.load(open(os.path.join(artifact_dir, "combo_info.json"))).get("quant", "?")
    except Exception:
        return "?"

# ── Discover result files ─────────────────────────────────────────────────────
# Artifact structure: <base>/<backend>/<model>/gen<G>/ctx<C>/profile_export_aiperf.json
results = []
for json_path in sorted(glob.glob(f"{base_dir}/**/profile_export_aiperf.json", recursive=True)):
    rel   = os.path.relpath(json_path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) < 5: continue
    backend    = parts[0]
    model_name = parts[1]
    gen = int(re.sub(r'\D', '', parts[2]))
    ctx = int(re.sub(r'\D', '', parts[3]))
    artifact_dir = os.path.dirname(json_path)
    try: d = json.load(open(json_path))
    except: continue
    def g(key, stat="avg"): return (d.get(key, {}) or {}).get(stat)
    avg_pw, avg_cpu, avg_gpu, peak_tj = parse_combo_power(artifact_dir)
    tps = g("output_token_throughput_per_user")
    tok_j = (tps / avg_pw) if (tps and avg_pw) else None
    results.append({
        "backend": backend, "model": model_name,
        "quant": get_combo_quant(artifact_dir),
        "prompt": ctx, "gen": gen,
        "isl": g("input_sequence_length"), "osl": g("output_sequence_length"),
        "osl_mis": g("osl_mismatch_diff_pct"),
        "ttft_avg": g("time_to_first_token"), "ttft_p50": g("time_to_first_token","p50"),
        "ttft_p90": g("time_to_first_token","p90"), "ttft_p99": g("time_to_first_token","p99"),
        "t2t_avg":  g("time_to_second_token"),"t2t_p50":  g("time_to_second_token","p50"),
        "t2t_p90":  g("time_to_second_token","p90"),"t2t_p99": g("time_to_second_token","p99"),
        "itl_avg":  g("inter_token_latency"), "itl_p50": g("inter_token_latency","p50"),
        "itl_p90":  g("inter_token_latency","p90"), "itl_p99": g("inter_token_latency","p99"),
        "tps": tps, "req_s": g("request_throughput"),
        "e2e_avg": g("e2e_output_token_throughput"),
        "e2e_p50": g("e2e_output_token_throughput","p50"),
        "e2e_p90": g("e2e_output_token_throughput","p90"),
        "e2e_p99": g("e2e_output_token_throughput","p99"),
        "rl_avg":  g("request_latency"),    "rl_p50":  g("request_latency","p50"),
        "rl_p90":  g("request_latency","p90"), "rl_p99": g("request_latency","p99"),
        "pre_avg": g("prefill_throughput_per_user"),
        "pre_p50": g("prefill_throughput_per_user","p50"),
        "pre_p90": g("prefill_throughput_per_user","p90"),
        "pre_p99": g("prefill_throughput_per_user","p99"),
        "power_w": avg_pw, "avg_cpu_c": avg_cpu, "avg_gpu_c": avg_gpu,
        "peak_tj_c": peak_tj, "tok_j": tok_j,
    })
results.sort(key=lambda r: (r["backend"], r["model"], r["gen"], r["prompt"]))

# ── Thermal summary: aggregate per-combo power across all combos per model ────
thermal_acc = {}
for r in results:
    k = f"{r['backend']}:{r['model']}"
    thermal_acc.setdefault(k, {"pw": [], "cpu": [], "gpu": [], "tj": []})
    if r["power_w"]   is not None: thermal_acc[k]["pw"].append(r["power_w"])
    if r["avg_cpu_c"] is not None: thermal_acc[k]["cpu"].append(r["avg_cpu_c"])
    if r["avg_gpu_c"] is not None: thermal_acc[k]["gpu"].append(r["avg_gpu_c"])
    if r["peak_tj_c"] is not None: thermal_acc[k]["tj"].append(r["peak_tj_c"])

thermal = {}
for k, v in thermal_acc.items():
    peak = max(v["tj"]) if v["tj"] else None
    thermal[k] = {
        "avg_pw":  sum(v["pw"])  / len(v["pw"])  if v["pw"]  else None,
        "avg_cpu": sum(v["cpu"]) / len(v["cpu"]) if v["cpu"] else None,
        "avg_gpu": sum(v["gpu"]) / len(v["gpu"]) if v["gpu"] else None,
        "peak_tj": peak,
        "throttled": peak is not None and peak > 85,
    }

best_tokj = {}
for r in results:
    k = f"{r['backend']}:{r['model']}"
    if r["tok_j"] is not None:
        if k not in best_tokj or r["tok_j"] > best_tokj[k]["tok_j"]: best_tokj[k] = r

lines = []
def L(s=""): lines.append(s)
def fmt(v, fs, fb="—"): return format(v, fs) if v is not None else fb

L("# MoE Single-Node Benchmark — 1× Jetson Orin Nano Super 8GB")
L()
L(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
L(f"**Backends:** {', '.join(sorted(set(r['backend'] for r in results)))}  ")
L(f"**Sweep:** prompt ∈ {{128,512,1024,2048}} tok × gen ∈ {{64,128,256}} tok  ")
L(f"**Artifacts:** `{base_dir}`")
L()

if skipped:
    L("## Skipped / Failed Models")
    for s in skipped: L(f"- {s}")
    L()

backends_present = sorted(set(r["backend"] for r in results))
for backend in backends_present:
    backend_results = [r for r in results if r["backend"] == backend]
    L(f"## Full Results — {backend}")
    L()
    L("> Power = VDD\\_CPU\\_GPU\\_CV avg over aiperf window.")
    L()
    H = ("| Model | Quant | ISL | OSL | OSL mis% "
         "| TTFT avg | p50 | p90 | p99 "
         "| T2T avg | p50 | p90 | p99 "
         "| ITL avg | p50 | p90 | p99 "
         "| Tok/s | Req/s "
         "| E2E avg | p50 | p90 | p99 "
         "| RL avg | p50 | p90 | p99 "
         "| Prefill avg | p50 | p90 | p99 "
         "| Power (W) | **Tok/J** |")
    S = ("|-------|:-----:|---:|---:|---:"
         "|---:|---:|---:|---:"
         "|---:|---:|---:|---:"
         "|---:|---:|---:|---:"
         "|---:|---:"
         "|---:|---:|---:|---:"
         "|---:|---:|---:|---:"
         "|---:|---:|---:|---:"
         "|---:|---:|")
    L(H); L(S)
    for r in backend_results:
        L(f"| {r['model']} | {r['quant']} "
          f"| {fmt(r['isl'],'.0f')} | {fmt(r['osl'],'.1f')} | {fmt(r['osl_mis'],'.2f')} "
          f"| {fmt(r['ttft_avg'],'.1f')} | {fmt(r['ttft_p50'],'.1f')} | {fmt(r['ttft_p90'],'.1f')} | {fmt(r['ttft_p99'],'.1f')} "
          f"| {fmt(r['t2t_avg'],'.2f')} | {fmt(r['t2t_p50'],'.2f')} | {fmt(r['t2t_p90'],'.2f')} | {fmt(r['t2t_p99'],'.2f')} "
          f"| {fmt(r['itl_avg'],'.2f')} | {fmt(r['itl_p50'],'.2f')} | {fmt(r['itl_p90'],'.2f')} | {fmt(r['itl_p99'],'.2f')} "
          f"| {fmt(r['tps'],'.2f')} | {fmt(r['req_s'],'.3f')} "
          f"| {fmt(r['e2e_avg'],'.2f')} | {fmt(r['e2e_p50'],'.2f')} | {fmt(r['e2e_p90'],'.2f')} | {fmt(r['e2e_p99'],'.2f')} "
          f"| {fmt(r['rl_avg'],'.1f')} | {fmt(r['rl_p50'],'.1f')} | {fmt(r['rl_p90'],'.1f')} | {fmt(r['rl_p99'],'.1f')} "
          f"| {fmt(r['pre_avg'],'.1f')} | {fmt(r['pre_p50'],'.1f')} | {fmt(r['pre_p90'],'.1f')} | {fmt(r['pre_p99'],'.1f')} "
          f"| {fmt(r['power_w'],'.2f')} | **{fmt(r['tok_j'],'.3f')}** |")
    L()

L("## Best Output Tok/J per Model")
L()
L("| Backend | Model | Quant | Best Tok/J | ISL | OSL | Tok/s | Power (W) |")
L("|---------|-------|:-----:|----------:|---:|---:|---:|---:|")
for key in sorted(best_tokj.keys()):
    b = best_tokj[key]
    L(f"| {b['backend']} | {b['model']} | {b['quant']} | **{fmt(b['tok_j'],'.3f')}** "
      f"| {fmt(b['isl'],'.0f')} | {fmt(b['osl'],'.1f')} "
      f"| {fmt(b['tps'],'.2f')} | {fmt(b['power_w'],'.2f')} |")

L()
L("## Thermal Summary")
L()
L("| Backend | Model | Avg Power (W) | Avg CPU (°C) | Avg GPU (°C) | Peak TJ (°C) | Throttled |")
L("|---------|-------|---:|---:|---:|---:|:---:|")
for key in sorted(thermal.keys()):
    t = thermal[key]
    backend, name = key.split(":", 1)
    L(f"| {backend} | {name} | {fmt(t['avg_pw'],'.2f')} | {fmt(t['avg_cpu'],'.1f')} "
      f"| {fmt(t['avg_gpu'],'.1f')} | {fmt(t['peak_tj'],'.1f')} "
      f"| {'YES ⚠' if t['throttled'] else 'No'} |")

L()
L("---")
L(f"*Generated by `benchmark-moe.sh` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
L(f"*{len(results)} rows  |  {len(set(r['backend'] for r in results))} backends  |  {len(set(r['model'] for r in results))} models*")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  Report → {report_path}")
print(f"  {len(results)} rows across {len(backends_present)} backend(s)")
PYEOF

    SKIPPED_STR=""
    for s in "${SKIPPED_MODELS[@]}"; do SKIPPED_STR="${SKIPPED_STR}${s}||"; done

    python3 "$REPORT_PY" "$BASE_ARTIFACT" "$SKIPPED_STR"

    banner "Done — $POWER_MODE_NAME"
    echo "  Artifacts : $BASE_ARTIFACT"
    echo "  Report    : $BASE_ARTIFACT/report.md"
    echo ""
done

deactivate 2>/dev/null || true

banner "All power modes complete"
echo "  Modes     : ${SWEEP_NAMES[*]}"
echo ""
echo "  Dashboard (optional):"
echo "    source ~/venv/bin/activate"
echo "    AIPERF_DASHBOARD_HOST=0.0.0.0 aiperf plot <artifact_dir> --dashboard --port 8050"
echo ""
