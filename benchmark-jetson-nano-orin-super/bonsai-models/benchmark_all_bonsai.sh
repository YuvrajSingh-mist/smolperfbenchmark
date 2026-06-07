#!/bin/bash
# benchmark_all_bonsai.sh
#
# Profiles all Bonsai + Ternary-Bonsai models on Jetson Orin 8GB.
# Backends: llamacpp (default) | ollama
# Per model: ONE server launch → smoke (32/256/512 tok) → if PASS, aiperf sweep.
# Sweeps: prompt_len in {256,512,1024,2048} x gen_len in {128,256,512}
# Key metric: tok/J (tokens per joule), derived from aiperf + tegrastats.
#
# Usage:
#   bash benchmark_all_bonsai.sh                          # llamacpp, MAXN_SUPER + clocks locked
#   bash benchmark_all_bonsai.sh --backend ollama         # Ollama backend
#   bash benchmark_all_bonsai.sh --power-mode 1           # run at 25W  (0=15W 1=25W 2=MAXN_SUPER 3=7W)
#   bash benchmark_all_bonsai.sh --no-lock-clocks
#   bash benchmark_all_bonsai.sh --reqs 5
#   bash benchmark_all_bonsai.sh --only bonsai-1.7b
#   bash benchmark_all_bonsai.sh --skip-download
#   bash benchmark_all_bonsai.sh --skip-smoke
#   bash benchmark_all_bonsai.sh --dry-run
#   bash benchmark_all_bonsai.sh --resume <dir>
#   bash benchmark_all_bonsai.sh --only Ternary-Bonsai-4B --resume artifacts/llamacpp/bonsai-all-20260527-0200-25W --power-mode 1  # rerun single missing model into existing run



set -e

# ── Auto-relaunch inside tmux if not already inside a session ─────────────────
if [ -z "$TMUX" ]; then
    SESSION="bonsai-bench"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "bash $(realpath "$0") $*" Enter
    exec tmux attach -t "$SESSION"
fi

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND="llamacpp"    # llamacpp | ollama
LLAMACPP_PORT=8080
OLLAMA_PORT=11434
SERVER_URL=""         # resolved after arg parsing

REQS=20
ONLY_MODEL=""
SKIP_DOWNLOAD=0
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
POWER_MODE=2          # 0=15W  1=25W  2=MAXN_SUPER  3=7W
LOCK_CLOCKS=1
CONCURRENCY=1
SLICE_DURATION=30
RANDOM_SEED=42
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10
COOLDOWN_MODEL=30
SERVER_STARTUP_TIMEOUT=300
CONTEXT_SIZE=2560     # max_prompt(2048) + max_gen(512)

PROMPT_LENGTHS=(256 512 1024 2048)
GEN_LENGTHS=(128 256 512)

SERVER_BIN="$SCRIPT_DIR/../Bonsai-demo/bin/cuda/llama-server"
HF_CLI="$HOME/venv/bin/hf"
BASE_ARTIFACT=""
TEGRA_PIDFILE="/tmp/bonsai_bench_tegrastats.pid"
SERVER_PIDFILE="/tmp/bonsai_bench_server.pid"
REPORT_PY="/tmp/bonsai_report.py"

# ── Model table: name|quant|gguf_path|tokenizer|hf_repo|hf_filename|ctx_size ──
declare -a MODELS=(
    "Bonsai-8B|Q1_0|$HOME/models/bonsai/8B/Bonsai-8B-Q1_0.gguf|Qwen/Qwen3-8B|prism-ml/Bonsai-8B-gguf|Bonsai-8B-Q1_0.gguf|2560"
    # "Ternary-Bonsai-8B|Q2_0|$HOME/models/ternary/8B/Ternary-Bonsai-8B-Q2_0.gguf|Qwen/Qwen3-8B|prism-ml/Ternary-Bonsai-8B-gguf|Ternary-Bonsai-8B-Q2_0.gguf|2560"
    "Ternary-Bonsai-4B|Q2_0|$HOME/models/ternary/4B/Ternary-Bonsai-4B-Q2_0.gguf|Qwen/Qwen3-4B|prism-ml/Ternary-Bonsai-4B-gguf|Ternary-Bonsai-4B-Q2_0.gguf|2560"
    "Bonsai-4B|Q1_0|$HOME/models/bonsai/4B/Bonsai-4B-Q1_0.gguf|Qwen/Qwen3-4B|prism-ml/Bonsai-4B-gguf|Bonsai-4B-Q1_0.gguf|2560"
    "Ternary-Bonsai-1.7B|Q2_0|$HOME/models/ternary/1.7B/Ternary-Bonsai-1.7B-Q2_0.gguf|Qwen/Qwen3-1.7B|prism-ml/Ternary-Bonsai-1.7B-gguf|Ternary-Bonsai-1.7B-Q2_0.gguf|2560"
    "Bonsai-1.7B|Q1_0|$HOME/models/bonsai/1.7B/Bonsai-1.7B-Q1_0.gguf|Qwen/Qwen3-1.7B|prism-ml/Bonsai-1.7B-gguf|Bonsai-1.7B-Q1_0.gguf|2560"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)        BACKEND="$2";       shift 2 ;;
        --reqs)           REQS="$2";          shift 2 ;;
        --only)           ONLY_MODEL="$2";    shift 2 ;;
        --skip-download)  SKIP_DOWNLOAD=1;    shift ;;
        --skip-smoke)     SKIP_SMOKE=1;       shift ;;
        --dry-run)        DRY_RUN=1;          shift ;;
        --resume)         RESUME_DIR="$2";    shift 2 ;;
        --power-mode)     POWER_MODE="$2";    shift 2 ;;
        --no-lock-clocks) LOCK_CLOCKS=0;      shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$BACKEND" in
    llamacpp) SERVER_URL="http://localhost:$LLAMACPP_PORT" ;;
    ollama)   SERVER_URL="http://localhost:$OLLAMA_PORT"   ;;
    *) echo "ERROR: unknown backend '$BACKEND' — use llamacpp or ollama"; exit 1 ;;
esac

# ── Resolve artifact dir ──────────────────────────────────────────────────────
if [ -n "$RESUME_DIR" ]; then
    [ -d "$RESUME_DIR" ] || { echo "ERROR: --resume dir not found: $RESUME_DIR"; exit 1; }
    BASE_ARTIFACT="$RESUME_DIR"
    SKIP_DOWNLOAD=1
    echo "  [RESUME] Reusing artifact dir: $BASE_ARTIFACT"
else
    BASE_ARTIFACT="$SCRIPT_DIR/artifacts/llamacpp/bonsai-${BACKEND}-$(date +%Y%m%d-%H%M)"
fi

TEGRA_LOG="$BASE_ARTIFACT/tegrastats.log"
TIMING_LOG="$BASE_ARTIFACT/model_timing.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
log()    { echo "  [$(date +%H:%M:%S)] $*"; }

# ── Initial cleanup ───────────────────────────────────────────────────────────
banner "Cleanup: killing stale processes before start"
if [ "$BACKEND" = "llamacpp" ]; then
    log "Stopping ollama (frees GPU/IOVA memory)..."
    sudo systemctl stop ollama 2>/dev/null && log "  ollama stopped" || log "  ollama not running"
    log "Killing any llama-server on :$LLAMACPP_PORT..."
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null && log "  killed llama-server" || log "  none found"
else
    log "Backend=ollama: stopping any llama-server (frees IOVA memory)..."
    sudo pkill -f "llama-server" 2>/dev/null && log "  killed llama-server" || log "  none found"
    log "Ensuring ollama daemon is running..."
    sudo systemctl start ollama 2>/dev/null || ollama serve > /tmp/ollama_serve.log 2>&1 &
    sleep 3
fi
log "Killing any tegrastats..."
sudo pkill -f "tegrastats" 2>/dev/null && log "  killed tegrastats" || log "  none found"
rm -f "$TEGRA_PIDFILE" "$SERVER_PIDFILE"
sleep 2
echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
CMA_FREE_KB=$(awk '/CmaFree/{print $2}' /proc/meminfo)
CMA_FREE_MIB=$(( CMA_FREE_KB / 1024 ))
log "CMA free: ${CMA_FREE_MIB} MiB / $(( $(awk '/CmaTotal/{print $2}' /proc/meminfo) / 1024 )) MiB total"

# ── Server management helpers ─────────────────────────────────────────────────

CURRENT_OLLAMA_TAG=""  # tracks which ollama model tag is currently loaded

kill_server() {
    if [ "$BACKEND" = "ollama" ]; then
        if [ -n "${CURRENT_OLLAMA_TAG:-}" ]; then
            log "  Unloading $CURRENT_OLLAMA_TAG from ollama..."
            ollama stop "$CURRENT_OLLAMA_TAG" 2>/dev/null || true
        fi
        sleep 8
        echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
        return
    fi
    # llamacpp
    if [ -f "$SERVER_PIDFILE" ]; then
        local pid; pid=$(cat "$SERVER_PIDFILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$SERVER_PIDFILE"
    fi
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null || true
    sleep 12
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
}

# For ollama: ping model health and reload if evicted between benchmark combos.
ensure_ollama_loaded() {
    local tag="$1"
    local code
    code=$(curl -s "$SERVER_URL/v1/models" --max-time 5 -o /dev/null -w "%{http_code}" 2>/dev/null || true)
    [ "$code" != "200" ] && { log "  [!] ollama daemon not responding (HTTP $code)"; return 1; }
    # Quick generation probe to confirm model is in memory
    set +e
    local resp
    resp=$(curl -s "$SERVER_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$tag\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
        --max-time 30 2>/dev/null)
    local ok=$?
    set -e
    if [ "$ok" = 0 ] && echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        return 0
    fi
    # Model evicted — reload
    log "  [!] Model evicted, reloading $tag..."
    curl -s "$SERVER_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$tag\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
        --max-time 120 -o /dev/null 2>/dev/null || true
    return 0
}

# Restart llamacpp server if it crashes mid-benchmark; noop check for ollama.
ensure_server_alive() {
    local model_path="$1" ctx_size="$2" srv_log="$3"
    if [ "$BACKEND" = "ollama" ]; then
        ensure_ollama_loaded "${CURRENT_OLLAMA_TAG:-}"
        return
    fi
    local code
    code=$(curl -s "$SERVER_URL/v1/models" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || true)
    [ "$code" = "200" ] && return 0
    log "  [!] Server not responding (HTTP $code) — restarting..."
    kill_server
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1 || true
    sleep 3
    "$SERVER_BIN" \
        -m "$model_path" \
        --host 0.0.0.0 --port "$LLAMACPP_PORT" \
        -ngl 99 --parallel 1 -c "$ctx_size" \
        --cache-ram 0 --no-cache-prompt --reasoning off \
        >> "$srv_log" 2>&1 &
    echo $! > "$SERVER_PIDFILE"
    log "  [RESTART] PID $(cat $SERVER_PIDFILE) — waiting for HTTP 200..."
    local elapsed=0
    while [ "$elapsed" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
        sleep 2; elapsed=$((elapsed + 2))
        if ! kill -0 "$(cat $SERVER_PIDFILE 2>/dev/null)" 2>/dev/null; then
            log "  [RESTART FAIL] Server died"; return 1
        fi
        code=$(curl -s "$SERVER_URL/v1/models" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        [ "$code" = "200" ] && { log "  [RESTART OK] t=${elapsed}s"; return 0; }
    done
    log "  [RESTART FAIL] Timed out after ${SERVER_STARTUP_TIMEOUT}s"
    return 1
}

stop_tegrastats() {
    [ -f "$TEGRA_PIDFILE" ] && sudo kill "$(cat $TEGRA_PIDFILE)" 2>/dev/null || true
    rm -f "$TEGRA_PIDFILE"
    sudo pkill -f "tegrastats" 2>/dev/null || true
}

# ── Phase 0: Parse model table ────────────────────────────────────────────────
declare -a MODEL_NAMES MODEL_QUANTS MODEL_PATHS MODEL_TOKENIZERS MODEL_DL_FAMILIES MODEL_DL_SIZES MODEL_CTX_SIZES

for entry in "${MODELS[@]}"; do
    IFS='|' read -r n q p t f s c <<< "$entry"
    MODEL_NAMES+=("$n"); MODEL_QUANTS+=("$q"); MODEL_PATHS+=("$p")
    MODEL_TOKENIZERS+=("$t"); MODEL_DL_FAMILIES+=("$f"); MODEL_DL_SIZES+=("$s")
    MODEL_CTX_SIZES+=("${c:-$CONTEXT_SIZE}")
done

banner "Bonsai All-Model Benchmark  |  Jetson Orin 8GB"
echo "  Date      : $(date --iso-8601=seconds)"
echo "  Backend   : $BACKEND  ($SERVER_URL)"
echo "  Requests  : $REQS per run"
echo "  Prompts   : ${PROMPT_LENGTHS[*]}"
echo "  Gen lens  : ${GEN_LENGTHS[*]}"
echo "  Context   : $CONTEXT_SIZE tokens"
echo "  Artifacts : $BASE_ARTIFACT"
[ -n "$ONLY_MODEL" ] && echo "  Filter    : $ONLY_MODEL"
[ "$DRY_RUN" = 1 ]   && echo "  Mode      : DRY RUN"
[ -n "$RESUME_DIR" ]  && echo "  Mode      : RESUME"

mkdir -p "$BASE_ARTIFACT"

# Validate backend prerequisites
if [ "$BACKEND" = "llamacpp" ] && [ ! -f "$SERVER_BIN" ]; then
    echo "ERROR: llama-server not found at $SERVER_BIN"
    echo "       Build with: cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87"
    exit 1
fi
if [ "$BACKEND" = "ollama" ]; then
    command -v ollama &>/dev/null || { echo "ERROR: ollama not found in PATH"; exit 1; }
    if [ "$DRY_RUN" = 0 ]; then
        log "Waiting for ollama daemon..."
        ELAPSED=0; CODE="000"
        while [ "$ELAPSED" -lt 30 ]; do
            CODE=$(curl -s "$SERVER_URL/api/tags" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || true)
            [ "$CODE" = "200" ] && { log "  [OK] ollama daemon ready"; break; }
            sleep 2; ELAPSED=$((ELAPSED + 2))
        done
        [ "$CODE" != "200" ] && { echo "ERROR: ollama daemon not responding after 30s"; exit 1; }
    fi
fi

# ── Phase 1: Download missing models ─────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = 0 ] && [ "$DRY_RUN" = 0 ]; then
    banner "Phase 1: Download missing models from HuggingFace (prism-ml)"
    for i in "${!MODEL_NAMES[@]}"; do
        path="${MODEL_PATHS[$i]}"; hf_repo="${MODEL_DL_FAMILIES[$i]}"; hf_file="${MODEL_DL_SIZES[$i]}"
        if [ -f "$path" ]; then
            log "${MODEL_NAMES[$i]} already present"
        else
            log "Downloading ${MODEL_NAMES[$i]} from $hf_repo/$hf_file..."
            mkdir -p "$(dirname "$path")"
            "$HF_CLI" download "$hf_repo" "$hf_file" --local-dir "$(dirname "$path")"
            log "${MODEL_NAMES[$i]} → $path"
        fi
    done
fi

# ── Phase 2: System setup ─────────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Phase 2: Lock clocks"
    sudo nvpmodel -m "$POWER_MODE" 2>/dev/null && log "nvpmodel -m $POWER_MODE OK" || log "nvpmodel not available"
    if [ "$LOCK_CLOCKS" = "1" ]; then
        sudo jetson_clocks 2>/dev/null && log "jetson_clocks OK" || log "jetson_clocks not available"
    else
        log "jetson_clocks skipped (--no-lock-clocks)"
    fi
fi

# ── Activate aiperf venv ──────────────────────────────────────────────────────
source "$HOME/venv/bin/activate" 2>/dev/null || \
source "$HOME/aiperf-env/bin/activate" 2>/dev/null || \
{ echo "ERROR: no aiperf venv found (~venv or ~aiperf-env)"; exit 1; }

# ── Phase 3: Start tegrastats ─────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Phase 3: Start tegrastats"
    stop_tegrastats
    sudo tegrastats --interval 500 --logfile "$TEGRA_LOG" &
    echo $! > "$TEGRA_PIDFILE"
    log "tegrastats PID $(cat $TEGRA_PIDFILE)  → $TEGRA_LOG"
fi

# ── Phase 4: Model loop ───────────────────────────────────────────────────────
banner "Phase 4: Model loop  [backend=$BACKEND]"

declare -a SKIPPED_MODELS SMOKE_PASSED_MODELS BENCH_MODELS

# Qwen3 ChatML template for Ollama Modelfile (all Bonsai models are Qwen3-based)
OLLAMA_TEMPLATE='{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
<|im_start|>assistant
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}{{ end }}'

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_QUANT="${MODEL_QUANTS[$i]}"
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
    MODEL_CTX_SIZE="${MODEL_CTX_SIZES[$i]}"
    MAX_PROMPT=$(( MODEL_CTX_SIZE - 512 ))

    # API model name differs per backend
    if [ "$BACKEND" = "ollama" ]; then
        MODEL_API_NAME="local-${MODEL_NAME,,}"   # e.g. local-bonsai-8b
        CURRENT_OLLAMA_TAG="$MODEL_API_NAME"
    else
        MODEL_API_NAME="$MODEL_NAME"
    fi

    [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue

    if [ "$DRY_RUN" = 1 ]; then
        log "[DRY RUN] $MODEL_NAME  backend=$BACKEND  api_name=$MODEL_API_NAME"
        [ -f "$MODEL_PATH" ] && BENCH_MODELS+=("$i")
        continue
    fi

    # Resume: count missing combos
    MISSING_COMBOS=0
    for G in "${GEN_LENGTHS[@]}"; do
        for P in "${PROMPT_LENGTHS[@]}"; do
            [ "$P" -gt "$((MODEL_CTX_SIZE - 512))" ] && continue
            [ ! -f "$BASE_ARTIFACT/${MODEL_NAME}/gen${G}/ctx${P}/profile_export_aiperf.json" ] && \
                MISSING_COMBOS=$((MISSING_COMBOS + 1))
        done
    done
    if [ "$MISSING_COMBOS" = 0 ]; then
        log "  [RESUME SKIP] $MODEL_NAME — all combos complete"
        BENCH_MODELS+=("$i"); continue
    fi
    [ -n "$RESUME_DIR" ] && log "  [RESUME] $MODEL_NAME — $MISSING_COMBOS combo(s) remaining"

    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    printf "  │  Model   : %-41s│\n" "$MODEL_NAME ($MODEL_QUANT)"
    printf "  │  Backend : %-41s│\n" "$BACKEND  →  $MODEL_API_NAME"
    printf "  │  Tok     : %-41s│\n" "$MODEL_TOKENIZER"
    printf "  │  Ctx     : %-41s│\n" "-c $MODEL_CTX_SIZE  (max prompt: $MAX_PROMPT tok)"
    echo "  └─────────────────────────────────────────────────────┘"

    if [ ! -f "$MODEL_PATH" ]; then
        log "  [SKIP] GGUF not found: $MODEL_PATH"
        SKIPPED_MODELS+=("$MODEL_NAME (file not found)"); continue
    fi

    SERVER_LOG="$BASE_ARTIFACT/${MODEL_NAME}-server.log"
    READY=0

    # ── Start / load model ────────────────────────────────────────────────────
    if [ "$BACKEND" = "llamacpp" ]; then
        log "Launching llama-server on :$LLAMACPP_PORT..."
        "$SERVER_BIN" \
            -m "$MODEL_PATH" \
            --host 0.0.0.0 --port "$LLAMACPP_PORT" \
            -ngl 99 --parallel 1 -c "$MODEL_CTX_SIZE" \
            --cache-ram 0 --no-cache-prompt --reasoning off \
            > "$SERVER_LOG" 2>&1 &
        echo $! > "$SERVER_PIDFILE"
        log "  PID: $(cat $SERVER_PIDFILE)  log: $SERVER_LOG"

        log "Waiting for HTTP 200..."
        ELAPSED=0
        while [ "$ELAPSED" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
            sleep 2; ELAPSED=$((ELAPSED + 2))
            if ! kill -0 "$(cat $SERVER_PIDFILE 2>/dev/null)" 2>/dev/null; then
                log "  [!] Server died at t=${ELAPSED}s (OOM?)"
                grep -E "error|OOM|failed|CUDA" "$SERVER_LOG" | tail -5 | sed 's/^/      /'
                kill_server; break
            fi
            CODE=$(curl -s "$SERVER_URL/v1/models" --max-time 3 -o /dev/null -w "%{http_code}" 2>/dev/null || true)
            if   [ "$CODE" = "200" ]; then READY=1; log "  [OK] HTTP 200 at t=${ELAPSED}s"; break
            elif [ "$CODE" = "503" ]; then log "  t=${ELAPSED}s: loading weights..."
            else                           log "  t=${ELAPSED}s: HTTP $CODE"; fi
        done

    else
        # ── Ollama: write Modelfile → ollama create → warmup ──────────────────
        log "Creating Modelfile for $MODEL_API_NAME..."
        cat > /tmp/bonsai_Modelfile << MODELEOF
FROM $MODEL_PATH
PARAMETER num_ctx $MODEL_CTX_SIZE
PARAMETER stop "<|im_end|>"
TEMPLATE """$OLLAMA_TEMPLATE"""
MODELEOF

        log "Registering with ollama (ollama create $MODEL_API_NAME)..."
        OLLAMA_OUT=$(ollama create "$MODEL_API_NAME" -f /tmp/bonsai_Modelfile 2>&1) && {
            log "  [OK] registered"
        } || {
            log "  [WARN] ollama create: $(echo "$OLLAMA_OUT" | tail -2 | tr -d '\n')"
        }

        log "Warming up $MODEL_API_NAME (loads weights into unified memory)..."
        set +e
        WARMUP_RESP=$(curl -s "$SERVER_URL/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL_API_NAME\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
            --max-time 120 2>/dev/null)
        WARMUP_EXIT=$?
        set -e

        if [ "$WARMUP_EXIT" = 0 ] && echo "$WARMUP_RESP" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
            READY=1; log "  [OK] model loaded into memory"
        else
            log "  [!] Warmup failed (exit=$WARMUP_EXIT): ${WARMUP_RESP:0:120}"
        fi
    fi

    if [ "$READY" = 0 ]; then
        log "  [FAIL] $MODEL_NAME — could not start/load, skipping"
        SKIPPED_MODELS+=("$MODEL_NAME (server/model failed to start)")
        kill_server; continue
    fi

    # ── Smoke test: 3 graded prompts ─────────────────────────────────────────
    if [ "$SKIP_SMOKE" = 1 ]; then
        log "  [SMOKE SKIP] --skip-smoke set"
        SMOKE_PASSED_MODELS+=("$MODEL_NAME (smoke skipped)")
    else
        log "  ── Smoke ────────────────────────────────────────────"
        SMOKE_PASS=1
        for tok_count in 32 256 512; do
            msg=$(python3 -c "print('The quick brown fox jumped over the lazy dog. ' * $((tok_count / 8 + 1)))")
            log "  Sending ~${tok_count}-tok prompt..."
            set +e
            resp=$(curl -s "$SERVER_URL/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL_API_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"$msg\"}],\"max_tokens\":32}" \
                --max-time 120 2>/dev/null)
            CURL_EXIT=$?
            set -e

            if [ "$BACKEND" = "llamacpp" ] && ! kill -0 "$(cat $SERVER_PIDFILE 2>/dev/null)" 2>/dev/null; then
                log "  [!] Server died after ~${tok_count}-tok prompt (OOM?)"
                grep -E "error|OOM|failed|CUDA|memory" "$SERVER_LOG" | tail -5 | sed 's/^/      /'
                SMOKE_PASS=0; break
            fi
            if [ "$CURL_EXIT" != 0 ]; then
                log "  [!] curl failed (exit $CURL_EXIT) at ~${tok_count} tok"
                SMOKE_PASS=0; break
            fi
            set +e
            echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('choices',[{}])[0].get('message')" 2>/dev/null
            PY_EXIT=$?
            set -e
            if [ "$PY_EXIT" = 0 ]; then
                log "  [OK] ~${tok_count} tok — valid response"
            else
                log "  [!] ~${tok_count} tok — bad response: ${resp:0:120}"
                SMOKE_PASS=0; break
            fi
        done
        log "  ─────────────────────────────────────────────────────"

        if [ "$SMOKE_PASS" = 0 ]; then
            log "  [SMOKE FAIL] $MODEL_NAME — skipping benchmark"
            SKIPPED_MODELS+=("$MODEL_NAME (smoke test failed)")
            kill_server; continue
        fi
        log "  [SMOKE PASS] $MODEL_NAME — starting aiperf sweep"
        SMOKE_PASSED_MODELS+=("$MODEL_NAME")
    fi
    BENCH_MODELS+=("$i")

    echo "MODEL_START:${MODEL_NAME}:$(date +%s)" >> "$TIMING_LOG"

    # ── aiperf sweep ─────────────────────────────────────────────────────────
    RUN_NUM=0
    VALID_PROMPTS=0
    for p in "${PROMPT_LENGTHS[@]}"; do [ "$p" -le "$MAX_PROMPT" ] && VALID_PROMPTS=$((VALID_PROMPTS+1)); done
    TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * VALID_PROMPTS ))

    for GEN in "${GEN_LENGTHS[@]}"; do
        for CTX in "${PROMPT_LENGTHS[@]}"; do
            if [ "$CTX" -gt "$MAX_PROMPT" ]; then
                log "  [SKIP] prompt=$CTX > max_prompt=$MAX_PROMPT"
                continue
            fi
            RUN_NUM=$((RUN_NUM + 1))
            ARTIFACT_DIR="$BASE_ARTIFACT/${MODEL_NAME}/gen${GEN}/ctx${CTX}"
            mkdir -p "$ARTIFACT_DIR"

            if [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ]; then
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  [RESUME SKIP]"
                continue
            fi

            log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"

            if ! ensure_server_alive "$MODEL_PATH" "$MODEL_CTX_SIZE" "$SERVER_LOG"; then
                log "  [ABORT] Cannot recover server for $MODEL_NAME"
                SKIPPED_MODELS+=("$MODEL_NAME (server unrecoverable at gen=$GEN ctx=$CTX)")
                break 2
            fi

            aiperf profile \
                --model                          "$MODEL_API_NAME" \
                --streaming \
                --endpoint-type                  'chat' \
                --url                            "$SERVER_URL" \
                --tokenizer                      "$MODEL_TOKENIZER" \
                --synthetic-input-tokens-mean    "$CTX" \
                --synthetic-input-tokens-stddev  0 \
                --output-tokens-mean             "$GEN" \
                --request-count                  "$REQS" \
                --concurrency                    "$CONCURRENCY" \
                --slice-duration                 "$SLICE_DURATION" \
                --random-seed                    "$RANDOM_SEED" \
                --request-timeout-seconds        "$REQUEST_TIMEOUT" \
                --artifact-dir                   "$ARTIFACT_DIR" \
                || log "  aiperf failed (ctx=$CTX gen=$GEN)"

            if [ "$RUN_NUM" -lt "$TOTAL_RUNS" ]; then
                log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"
            fi
        done
    done

    echo "MODEL_END:${MODEL_NAME}:$(date +%s)" >> "$TIMING_LOG"
    kill_server
    log "Model done. Cooldown ${COOLDOWN_MODEL}s before next model..."
    sleep "$COOLDOWN_MODEL"
done

deactivate 2>/dev/null || true

# ── Phase 5: Stop tegrastats ──────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Phase 5: Stop tegrastats"
    stop_tegrastats
    log "tegrastats stopped"
fi

if [ "$DRY_RUN" = 1 ]; then
    banner "Dry run complete"; exit 0
fi

# ── Phase 6: Summary ──────────────────────────────────────────────────────────
banner "Phase 6: Smoke test results"
echo ""
echo "  PASSED  (${#SMOKE_PASSED_MODELS[@]})"
for m in "${SMOKE_PASSED_MODELS[@]}"; do echo "    [PASS] $m"; done
echo ""
echo "  FAILED / SKIPPED  (${#SKIPPED_MODELS[@]})"
for m in "${SKIPPED_MODELS[@]}"; do echo "    [FAIL] $m"; done
echo ""

if [ "${#BENCH_MODELS[@]}" = 0 ]; then
    echo "  No models passed smoke test — no report generated."; exit 1
fi

# ── Phase 7: Generate report.md ───────────────────────────────────────────────
banner "Phase 7: Generating report.md"

cat > "$REPORT_PY" << 'PYEOF'
import re, sys, json, os, glob
from datetime import datetime

base_dir    = sys.argv[1]
tegra_log   = sys.argv[2]
timing_log  = sys.argv[3]
skipped_arg = sys.argv[4] if len(sys.argv) > 4 else ""
ctx_size    = sys.argv[5] if len(sys.argv) > 5 else "?"
backend     = sys.argv[6] if len(sys.argv) > 6 else "llamacpp"
report_path = os.path.join(base_dir, "report.md")

skipped = [s for s in skipped_arg.split("||") if s] if skipped_arg else []

class PowerSample:
    def __init__(self, ts, pw, cpu_t, gpu_t, tj_t):
        self.ts=ts; self.pw=pw; self.cpu_t=cpu_t; self.gpu_t=gpu_t; self.tj_t=tj_t

samples = []
try:
    for line in open(tegra_log):
        ts_m = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
        if not ts_m: continue
        try: ts = datetime.strptime(ts_m.group(1), "%m-%d-%Y %H:%M:%S").timestamp()
        except: continue
        pw_m  = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
        cpu_m = re.search(r'cpu@([\d.]+)C', line)
        gpu_m = re.search(r'gpu@([\d.]+)C', line)
        tj_m  = re.search(r'tj@([\d.]+)C',  line)
        if pw_m:
            samples.append(PowerSample(ts, int(pw_m.group(1))/1000.0,
                float(cpu_m.group(1)) if cpu_m else None,
                float(gpu_m.group(1)) if gpu_m else None,
                float(tj_m.group(1))  if tj_m  else None))
except FileNotFoundError:
    pass

def power_in_window(t0, t1):
    w = [s for s in samples if t0 <= s.ts <= t1]
    if not w: return None, None, None, None
    avg_pw = sum(s.pw for s in w) / len(w)
    cpu_t = [s.cpu_t for s in w if s.cpu_t is not None]
    gpu_t = [s.gpu_t for s in w if s.gpu_t is not None]
    tj_t  = [s.tj_t  for s in w if s.tj_t  is not None]
    return (avg_pw,
            sum(cpu_t)/len(cpu_t) if cpu_t else None,
            sum(gpu_t)/len(gpu_t) if gpu_t else None,
            max(tj_t)             if tj_t  else None)

model_windows = {}
try:
    for line in open(timing_log):
        line = line.strip()
        if line.startswith("MODEL_START:"):
            _, name, ts = line.split(":", 2)
            model_windows.setdefault(name, {})["start"] = float(ts)
        elif line.startswith("MODEL_END:"):
            _, name, ts = line.split(":", 2)
            model_windows.setdefault(name, {})["end"] = float(ts)
except FileNotFoundError:
    pass

results = []
for json_path in sorted(glob.glob(f"{base_dir}/**/profile_export_aiperf.json", recursive=True)):
    rel   = os.path.relpath(json_path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) < 4: continue
    model_name = parts[0]
    gen = int(re.sub(r'\D', '', parts[1]))
    ctx = int(re.sub(r'\D', '', parts[2]))
    try: d = json.load(open(json_path))
    except: continue
    ttft = (d.get("time_to_first_token",              {}) or {}).get("avg")
    itl  = (d.get("inter_token_latency",              {}) or {}).get("avg")
    tps  = (d.get("output_token_throughput_per_user", {}) or {}).get("avg")
    rl   = (d.get("request_latency",                  {}) or {}).get("avg")
    win  = model_windows.get(model_name, {})
    avg_pw, avg_cpu, avg_gpu, peak_tj = power_in_window(win.get("start",0), win.get("end",9e18))
    tok_j = (tps / avg_pw) if (tps and avg_pw) else None
    quant = "Q2_0" if "Ternary" in model_name else "Q1_0"
    bits  = "1.58-bit" if quant == "Q2_0" else "1-bit"
    results.append({"model": model_name, "quant": quant, "bits": bits,
        "prompt": ctx, "gen": gen,
        "ttft": ttft, "itl": itl, "tps": tps, "req_lat": rl,
        "power_w": avg_pw, "avg_cpu_c": avg_cpu, "avg_gpu_c": avg_gpu,
        "peak_tj_c": peak_tj, "tok_j": tok_j})

results.sort(key=lambda r: (r["model"], r["gen"], r["prompt"]))

thermal = {}
for model_name, win in model_windows.items():
    avg_pw, avg_cpu, avg_gpu, peak_tj = power_in_window(win.get("start",0), win.get("end",9e18))
    thermal[model_name] = {"avg_pw": avg_pw, "avg_cpu": avg_cpu,
        "avg_gpu": avg_gpu, "peak_tj": peak_tj,
        "throttled": peak_tj is not None and peak_tj > 85}

best_tokj = {}
for r in results:
    m = r["model"]
    if r["tok_j"] is not None:
        if m not in best_tokj or r["tok_j"] > best_tokj[m]["tok_j"]: best_tokj[m] = r

lines = []
def L(s=""): lines.append(s)
def fmt(v, fmt_str, fallback="—"):
    return format(v, fmt_str) if v is not None else fallback

L("# Bonsai All-Model Benchmark — Jetson Orin Nano Super 8GB")
L()
L(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
L(f"**Backend:** {backend}  **Context:** {ctx_size} tokens  **Concurrency:** 1  ")
L(f"**Sweep:** prompt in {{256,512,1024,2048}}  gen in {{128,256,512}}")
L(f"**Artifacts:** `{base_dir}`")
L()

if skipped:
    L("## Skipped Models")
    L()
    for s in skipped: L(f"- {s}")
    L()

L("## Full Results")
L()
L("| Model | Bits | Prompt (tok) | Gen (tok) | TTFT avg (ms) | ITL avg (ms) | Tok/s | Power (W) | **Tok/J** |")
L("|-------|:----:|:---:|:---:|---:|---:|---:|---:|---:|")
for r in results:
    L(f"| {r['model']} | {r['bits']} | {r['prompt']} | {r['gen']} "
      f"| {fmt(r['ttft'], '.0f')} "
      f"| {fmt(r['itl'], '.2f')} "
      f"| {fmt(r['tps'], '.2f')} "
      f"| {fmt(r['power_w'], '.2f')} "
      f"| **{fmt(r['tok_j'], '.4f')}** |")

L()
L("## Per-Model Best Tok/J")
L()
L("| Model | Bits | Best Tok/J | Prompt (tok) | Gen (tok) | Tok/s | Power (W) |")
L("|-------|:----:|---:|:---:|:---:|---:|---:|")
for model_name in sorted(best_tokj.keys()):
    b = best_tokj[model_name]
    L(f"| {b['model']} | {b['bits']} | **{fmt(b['tok_j'], '.4f')}** "
      f"| {b['prompt']} | {b['gen']} "
      f"| {fmt(b['tps'], '.2f')} | {fmt(b['power_w'], '.2f')} |")

L()
L("## Thermal Summary")
L()
L("| Model | Avg Power (W) | Avg CPU (C) | Avg GPU (C) | Peak TJ (C) | Throttled |")
L("|-------|---:|---:|---:|---:|:---:|")
for model_name in sorted(thermal.keys()):
    t = thermal[model_name]
    L(f"| {model_name} "
      f"| {fmt(t['avg_pw'], '.2f')} "
      f"| {fmt(t['avg_cpu'], '.1f')} "
      f"| {fmt(t['avg_gpu'], '.1f')} "
      f"| {fmt(t['peak_tj'], '.1f')} "
      f"| {'YES' if t['throttled'] else 'No'} |")

L()
L("---")
L(f"*Generated by `benchmark_all_bonsai.sh` ({backend}) on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  Report -> {report_path}")
print(f"  {len(results)} rows  |  {len(thermal)} models  |  {len(skipped)} skipped")
PYEOF

SKIPPED_STR=""
for s in "${SKIPPED_MODELS[@]}"; do SKIPPED_STR="${SKIPPED_STR}${s}||"; done

python3 "$REPORT_PY" \
    "$BASE_ARTIFACT" "$TEGRA_LOG" "$TIMING_LOG" \
    "$SKIPPED_STR" "$CONTEXT_SIZE" "$BACKEND"

banner "Done"
echo "  Backend   : $BACKEND"
echo "  Artifacts : $BASE_ARTIFACT"
echo "  Report    : $BASE_ARTIFACT/report.md"
echo ""
echo "  Dashboard (optional):"
echo "    source ~/venv/bin/activate"
echo "    AIPERF_DASHBOARD_HOST=0.0.0.0 aiperf plot $BASE_ARTIFACT --dashboard --port 8050"
echo ""
