#!/bin/bash
# benchmark_non_reasoning.sh — LLM benchmark: Mac Mini M4 16GB
#
# Per model: ONE server launch → smoke → aiperf sweep.
# Sweeps: prompt in {256,512,1024,2048,4096} × gen in {256,512,1024}
# Power measured via macOS powermetrics (requires sudo; skip with --no-power)
#
# Usage:
#   bash benchmark_non_reasoning.sh                    # llamacpp only, all models
#   bash benchmark_non_reasoning.sh --backend ollama   # ollama only
#   bash benchmark_non_reasoning.sh --backend both     # llamacpp then ollama
#   bash benchmark_non_reasoning.sh --reqs 20
#   bash benchmark_non_reasoning.sh --only qwen3-8b
#   bash benchmark_non_reasoning.sh --skip-smoke
#   bash benchmark_non_reasoning.sh --dry-run
#   bash benchmark_non_reasoning.sh --resume DIR
#   bash benchmark_non_reasoning.sh --no-power         # skip powermetrics (no sudo needed)

set -euo pipefail

export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

# ── Ensure tmux is installed ──────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
    echo "tmux not found — installing via brew..."
    brew install tmux
fi

# ── Ensure hf CLI is available ────────────────────────────────────────────────
# huggingface_hub ≥1.0 ships the CLI as `hf` (not `huggingface-cli`).
VENV_PYTHON="$HOME/venv/bin/python3"
VENV_HF_CLI="$HOME/venv/bin/hf"
if [ ! -f "$VENV_HF_CLI" ]; then
    echo "hf CLI not found — installing huggingface_hub into ~/venv..."
    "$VENV_PYTHON" -m pip install -U "huggingface_hub"
fi
if [ ! -f "$VENV_HF_CLI" ]; then
    echo "ERROR: hf CLI not available after install."
    echo "  Try: $VENV_PYTHON -m pip install -U huggingface_hub"
    exit 1
fi

# ── Auto-relaunch inside tmux if not already there ────────────────────────────
if [ -z "${TMUX:-}" ]; then
    SESSION="non-reasoning-bench"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "bash $(realpath "$0") $(printf '%q ' "$@"); echo 'Done — press Enter to exit'; read"
    echo "Launched in tmux session '$SESSION'"
    echo "Attach with:  tmux attach -t $SESSION"
    exit 0
fi

# ── Cache sudo for the full run (powermetrics needs it) ───────────────────────
if [ "${NO_POWER:-0}" != 1 ]; then
    sudo -v
    ( while true; do sudo -n -v; sleep 60; done ) &
    SUDO_KEEPALIVE_PID=$!
    trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT
fi

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND="llamacpp"    # llamacpp | ollama | both
REQS=20
ONLY_MODEL=""
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
NO_POWER=0
CONCURRENCY=1
SLICE_DURATION=30
RANDOM_SEED=42
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10
COOLDOWN_MODEL=30
COOLDOWN_BACKEND=45
SERVER_STARTUP_TIMEOUT=300

PROMPT_LENGTHS=(256 512 1024 2048 4096)
GEN_LENGTHS=(256 512 1024)
CONTEXT_SIZE=6144   # max_prompt(4096) + max_gen(1024) + 1024 headroom

LLAMACPP_BIN="${LLAMACPP_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
LLAMACPP_PORT=8080
OLLAMA_PORT=11434

GGUF_DIR="$HOME/gguf-models"
POWER_PIDFILE="/tmp/blog_bench_power.pid"
RSS_PIDFILE="/tmp/blog_bench_rss.pid"
SERVER_PIDFILE="/tmp/blog_bench_server.pid"
REPORT_PY="/tmp/blog_report_mac.py"

# ── Model table: name|quant|gguf_path|tokenizer|ctx_size ─────────────────────
# GGUF source notes (searched June 2025):
#   Granite  → ibm-granite org publishes official GGUFs; 4.0-tiny is MoE so using 4.1 (dense)
#   Nemotron → NVIDIA does NOT publish GGUFs; bartowski is the canonical community source
#   Qwen     → Qwen org publishes official GGUFs for all three models
#   Gemma 3  → Google publishes QAT Q4_0 only; ggml-org (llama.cpp team) has Q4_K_M for 4B/9B/12B
declare -a MODELS=(
    "granite41-3b|Q4_K_M|$GGUF_DIR/granite41-3b-q4_k_m.gguf|ibm-granite/granite-4.1-3b|6144"
    "granite41-8b|Q4_K_M|$GGUF_DIR/granite41-8b-q4_k_m.gguf|ibm-granite/granite-4.1-8b|6144"
    "nemotron-mini-4b|Q4_K_M|$GGUF_DIR/nemotron-mini-4b-q4_k_m.gguf|nvidia/Nemotron-Mini-4B-Instruct|4096"
    "nemotron-nano-8b|Q4_K_M|$GGUF_DIR/nemotron-nano-8b-q4_k_m.gguf|nvidia/Llama-3.1-Nemotron-Nano-8B-v1|6144"
    "qwen3-4b|Q4_K_M|$GGUF_DIR/Qwen3-4B-Q4_K_M.gguf|Qwen/Qwen3-4B|6144"
    "qwen3-8b|Q4_K_M|$GGUF_DIR/Qwen3-8B-Q4_K_M.gguf|Qwen/Qwen3-8B|6144"
    "qwen2.5-7b|Q4_K_M|$GGUF_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf|Qwen/Qwen2.5-7B-Instruct|6144"
    "gemma3-4b|Q4_K_M|$GGUF_DIR/gemma3-4b-q4_k_m.gguf|google/gemma-3-4b-it|6144"
    "gemma3-12b|Q4_K_M|$GGUF_DIR/gemma3-12b-q4_k_m.gguf|google/gemma-3-12b-it|6144"
)

# ── GGUF download sources: local_filename -> "hf_repo hf_filename" ────────────
declare -A GGUF_SOURCES=(
    # IBM Granite 4.1 — official ibm-granite org GGUFs (dense, not MoE)
    ["granite41-3b-q4_k_m.gguf"]="ibm-granite/granite-4.1-3b-GGUF granite-4.1-3b-Q4_K_M.gguf"
    ["granite41-8b-q4_k_m.gguf"]="ibm-granite/granite-4.1-8b-GGUF granite-4.1-8b-Q4_K_M.gguf"

    # NVIDIA Nemotron — no official NVIDIA GGUF repo exists; bartowski is canonical
    ["nemotron-mini-4b-q4_k_m.gguf"]="bartowski/Nemotron-Mini-4B-Instruct-GGUF Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"
    # bartowski prefixes the file with "nvidia_" — confirmed June 2026
    ["nemotron-nano-8b-q4_k_m.gguf"]="bartowski/nvidia_Llama-3.1-Nemotron-Nano-8B-v1-GGUF nvidia_Llama-3.1-Nemotron-Nano-8B-v1-Q4_K_M.gguf"

    # Qwen — Qwen3 from official Qwen org; Qwen2.5-7B from bartowski (Qwen org only has split files)
    ["Qwen3-4B-Q4_K_M.gguf"]="Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf"
    ["Qwen3-8B-Q4_K_M.gguf"]="Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf"
    ["Qwen2.5-7B-Instruct-Q4_K_M.gguf"]="bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf"

    # Gemma 3 — ggml-org (llama.cpp maintainers) publishes Q4_K_M; Google only publishes QAT Q4_0
    # Note: Gemma 3 has no 9B variant; available sizes are 1b, 4b, 12b, 27b
    ["gemma3-4b-q4_k_m.gguf"]="ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf"
    ["gemma3-12b-q4_k_m.gguf"]="ggml-org/gemma-3-12b-it-GGUF gemma-3-12b-it-Q4_K_M.gguf"
)

# ── Ollama per-model config: name -> "template_type" ─────────────────────────
# template_type: chatml | granite | llama3 | gemma
declare -A OLLAMA_CFG=(
    # IBM Granite 4.1: custom role-based format with <|start_of_role|> tokens
    ["granite41-3b"]="granite"
    ["granite41-8b"]="granite"

    # NVIDIA Nemotron-Mini 4B: standard ChatML
    ["nemotron-mini-4b"]="chatml"

    # NVIDIA Nemotron-Nano 8B: Llama 3.1 base, uses llama3 header format
    ["nemotron-nano-8b"]="llama3"

    # Qwen3 + Qwen2.5: standard ChatML, EOS=<|im_end|>
    ["qwen3-4b"]="chatml"
    ["qwen3-8b"]="chatml"
    ["qwen2.5-7b"]="chatml"

    # Gemma 3: no system role; BOS added by tokenizer; stop on turn tokens
    ["gemma3-4b"]="gemma"
    ["gemma3-12b"]="gemma"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)    BACKEND="$2";     shift 2 ;;
        --reqs)       REQS="$2";        shift 2 ;;
        --only)       ONLY_MODEL="$2";  shift 2 ;;
        --skip-smoke) SKIP_SMOKE=1;     shift ;;
        --dry-run)    DRY_RUN=1;        shift ;;
        --resume)     RESUME_DIR="$2";  shift 2 ;;
        --no-power)   NO_POWER=1;       shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ "$BACKEND" =~ ^(llamacpp|ollama|both)$ ]] || \
    { echo "ERROR: --backend must be llamacpp|ollama|both"; exit 1; }

# ── Sweep setup ───────────────────────────────────────────────────────────────
SWEEP_DATE=$(date +%Y%m%d-%H%M)
if [ -n "$RESUME_DIR" ]; then
    BASE_ARTIFACT="$RESUME_DIR"
else
    BASE_ARTIFACT="$HOME/Desktop/smolbenchmark/benchmark-mac-mini-m4/artifacts/mac-m4-${SWEEP_DATE}"
fi
TIMING_LOG="$BASE_ARTIFACT/model_timing.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
log()    { echo "  [$(date +%H:%M:%S)] $*"; }

# ── Power monitoring via macOS powermetrics ───────────────────────────────────
start_combo_powermetrics() {
    local artifact_dir="$1"
    [ "$NO_POWER" = 1 ] && return
    [ -f "$POWER_PIDFILE" ] && sudo kill "$(cat "$POWER_PIDFILE")" 2>/dev/null || true
    rm -f "$POWER_PIDFILE"
    [ "$DRY_RUN" = 1 ] && return
    sudo powermetrics --samplers cpu_power,gpu_power,thermal \
        -i 50 -n 99999 \
        > "$artifact_dir/powermetrics.log" 2>/dev/null &
    echo $! > "$POWER_PIDFILE"
}

stop_combo_powermetrics() {
    [ "$NO_POWER" = 1 ] && return
    if [ -f "$POWER_PIDFILE" ]; then
        sudo kill "$(cat "$POWER_PIDFILE")" 2>/dev/null || true
        rm -f "$POWER_PIDFILE"
    fi
    sudo pkill -f "powermetrics" 2>/dev/null || true
}

# ── RSS sampling — peak physical RAM used by inference process per combo ──────
# Samples the target PID's RSS (KB) every 50 ms, writes one value per line.
# Report reads the max — that's peak model+KV-cache footprint during the run.
start_combo_rss() {
    local artifact_dir="$1" target_pid="$2"
    [ "$DRY_RUN" = 1 ] && return
    [ -f "$RSS_PIDFILE" ] && kill "$(cat "$RSS_PIDFILE")" 2>/dev/null || true
    rm -f "$RSS_PIDFILE"
    (
        while kill -0 "$target_pid" 2>/dev/null; do
            ps -o rss= -p "$target_pid" 2>/dev/null || true
            sleep 0.05
        done
    ) > "$artifact_dir/rss.log" &
    echo $! > "$RSS_PIDFILE"
}

stop_combo_rss() {
    [ -f "$RSS_PIDFILE" ] && kill "$(cat "$RSS_PIDFILE")" 2>/dev/null || true
    rm -f "$RSS_PIDFILE"
}

kill_llamacpp() {
    if [ -f "$SERVER_PIDFILE" ]; then
        kill "$(cat "$SERVER_PIDFILE")" 2>/dev/null || true
        rm -f "$SERVER_PIDFILE"
    fi
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null || true
    sleep 5
}

restart_ollama() {
    local model_tag="${1:-}"
    log "  [!] Attempting Ollama restart..."
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 2
    OLLAMA_HOST=0.0.0.0 OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=1 ollama serve &>/dev/null &
    sleep 5
    local elapsed=0
    while [ "$elapsed" -lt 60 ]; do
        local code
        code=$(curl -s "http://localhost:$OLLAMA_PORT/api/tags" --max-time 3 \
               -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        [ "$code" = "200" ] && log "  [OK] Ollama restarted at t=${elapsed}s" && break
        sleep 2; elapsed=$((elapsed + 2))
    done
    [ "$elapsed" -ge 60 ] && { log "  [FAIL] Ollama did not come back within 60s"; return 1; }
    if [ -n "$model_tag" ]; then
        log "  Re-warming $model_tag..."
        curl -s "http://localhost:$OLLAMA_PORT/api/generate" \
            -d "{\"model\":\"$model_tag\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
            --max-time 120 -o /dev/null 2>/dev/null || true
    fi
    return 0
}

ensure_llamacpp_alive() {
    local model_path="$1" ctx_size="$2" srv_log="$3"
    local code
    code=$(curl -s "http://localhost:$LLAMACPP_PORT/v1/models" --max-time 3 \
           -o /dev/null -w "%{http_code}" 2>/dev/null || true)
    [ "$code" = "200" ] && return 0

    log "  [!] llama-server not responding (HTTP $code) — restarting..."
    kill_llamacpp
    sleep 3
    "$LLAMACPP_BIN" -m "$model_path" --host 0.0.0.0 --port "$LLAMACPP_PORT" \
        -ngl 99 --parallel 1 -c "$ctx_size" \
        -t 1 -fa 1 --prio 2 --mlock \
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

# ── Smoke test ────────────────────────────────────────────────────────────────
smoke_test() {
    local url="$1" model_tag="$2"
    log "  ── Smoke ─────────────────────────────────────────────"
    local pass=1
    for tok_count in 32 256 512; do
        log "  Sending ~${tok_count}-tok prompt..."
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

# ── Ollama: build Modelfile content ──────────────────────────────────────────
make_ollama_modelfile() {
    local gguf_path="$1" tmpl_type="$2" ctx_size="$3"

    printf 'FROM %s\nPARAMETER num_ctx %s\nPARAMETER num_keep -1\n' \
        "$gguf_path" "$ctx_size"

    case "$tmpl_type" in

        # ── ChatML (Qwen2.5, Qwen3, Nemotron-Mini) ───────────────────────────
        chatml)
            cat <<'TMPL'
PARAMETER stop "<|im_end|>"
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
TMPL
            ;;

        # ── IBM Granite 4.0 ───────────────────────────────────────────────────
        # Uses <|start_of_role|>/<|end_of_role|> role delimiters, EOS=<|end_of_text|>
        # Source: ibm-granite/granite-4.0-tiny-preview-4k-instruct tokenizer_config.json
        granite)
            cat <<'TMPL'
PARAMETER stop "<|end_of_text|>"
TEMPLATE """{{ if .System }}<|start_of_role|>system<|end_of_role|>
{{ .System }}<|end_of_text|>
{{ end }}{{ if .Prompt }}<|start_of_role|>user<|end_of_role|>
{{ .Prompt }}<|end_of_text|>
{{ end }}<|start_of_role|>assistant<|end_of_role|>
{{ .Response }}<|end_of_text|>"""
TMPL
            ;;

        # ── Llama 3 (Nemotron-Nano 8B) ────────────────────────────────────────
        # BOS=<|begin_of_text|> added by tokenizer.
        # Source: meta-llama/Llama-3.1 tokenizer_config.json
        llama3)
            cat <<'TMPL'
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
TMPL
            ;;

        # ── Gemma 3 ───────────────────────────────────────────────────────────
        # No system role; BOS=<bos> added by tokenizer; role name is "model".
        # Source: google/gemma-3-4b-it tokenizer_config.json
        gemma)
            cat <<'TMPL'
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
TEMPLATE """{{ if .Prompt }}<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ end }}{{ .Response }}<end_of_turn>"""
TMPL
            ;;

        *)
            log "  [ERROR] Unknown template type: $tmpl_type"
            return 1
            ;;
    esac
}

# ── Ollama: import GGUF ───────────────────────────────────────────────────────
import_ollama_model() {
    local model_name="$1" gguf_path="$2" tmpl_type="$3" ctx_size="$4"
    local import_tag="local-${model_name}"

    if [ ! -f "$gguf_path" ]; then
        log "  [FAIL] GGUF not found: $gguf_path" >&2
        return 1
    fi

    log "  Importing GGUF as ${import_tag}..." >&2
    local tmp_modelfile="/tmp/ollama_modelfile_${model_name}"
    make_ollama_modelfile "$gguf_path" "$tmpl_type" "$ctx_size" > "$tmp_modelfile"
    local err
    if err=$(ollama create "$import_tag" -f "$tmp_modelfile" 2>&1); then
        log "  [OK] Imported as $import_tag" >&2
        echo "$import_tag"
        return 0
    else
        log "  [FAIL] GGUF import failed: $(echo "$err" | head -2 | tr '\n' ' ')" >&2
        return 1
    fi
}

# ── Parse model table ─────────────────────────────────────────────────────────
declare -a MODEL_NAMES MODEL_QUANTS MODEL_PATHS MODEL_TOKENIZERS MODEL_CTX_SIZES

for entry in "${MODELS[@]}"; do
    IFS='|' read -r n q p t c <<< "$entry"
    MODEL_NAMES+=("$n")
    MODEL_QUANTS+=("$q")
    MODEL_PATHS+=("$p")
    MODEL_TOKENIZERS+=("$t")
    MODEL_CTX_SIZES+=("${c:-$CONTEXT_SIZE}")
done

# ── Print config banner ───────────────────────────────────────────────────────
banner "LLM Benchmark  |  Mac Mini M4 16GB"
echo "  Date      : $(date -Iseconds)"
echo "  Backend   : $BACKEND"
echo "  Requests  : $REQS per run"
echo "  Prompts   : ${PROMPT_LENGTHS[*]}"
echo "  Gen lens  : ${GEN_LENGTHS[*]}"
echo "  Artifacts : $BASE_ARTIFACT"
[ -n "$ONLY_MODEL" ] && echo "  Filter    : $ONLY_MODEL"
[ "$DRY_RUN"  = 1  ] && echo "  Mode      : DRY RUN"
[ "$NO_POWER" = 1  ] && echo "  Power     : DISABLED"
[ -n "$RESUME_DIR" ] && echo "  Mode      : RESUME ($RESUME_DIR)"

# ── Initial cleanup ───────────────────────────────────────────────────────────
banner "Cleanup"
pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null && log "killed llama-server" || true
stop_combo_powermetrics 2>/dev/null || true
rm -f "$SERVER_PIDFILE"
sleep 2

# ── Pre-flight: download missing GGUFs ───────────────────────────────────────
banner "Pre-flight: checking / downloading missing GGUFs"
mkdir -p "$GGUF_DIR"
for local_name in "${!GGUF_SOURCES[@]}"; do
    local_path="$GGUF_DIR/$local_name"
    if [ -f "$local_path" ]; then
        log "  [OK]   $local_name"
        continue
    fi
    read -r hf_repo hf_file <<< "${GGUF_SOURCES[$local_name]}"
    log "  [DL]   $local_name  ←  $hf_repo / $hf_file"
    if "$VENV_HF_CLI" download "$hf_repo" "$hf_file" --local-dir "$GGUF_DIR"; then
        tmp_path="$GGUF_DIR/$hf_file"
        [ "$tmp_path" != "$local_path" ] && [ -f "$tmp_path" ] && mv "$tmp_path" "$local_path"
        [ -f "$local_path" ] && log "  [DONE] $local_name" || log "  [FAIL] $local_name not found after download"
    else
        log "  [FAIL] Could not download $local_name — model will be skipped"
    fi
done

# ── Activate aiperf venv ──────────────────────────────────────────────────────
AIPERF_BIN="$HOME/Desktop/smolbenchmark/venv/bin/aiperf"
[ -f "$AIPERF_BIN" ] || { echo "ERROR: aiperf not found at $AIPERF_BIN"; echo "  Run: pip install 'aiperf @ git+https://github.com/ai-dynamo/aiperf.git' inside ~/Desktop/smolbenchmark/venv/"; exit 1; }
source "$HOME/Desktop/smolbenchmark/venv/bin/activate"

mkdir -p "$BASE_ARTIFACT"

SKIPPED_MODELS=()
BENCH_MODELS=()

# ══════════════════════════════════════════════════════════════════════════════
# BACKEND: llama.cpp
# ══════════════════════════════════════════════════════════════════════════════
run_llamacpp() {
    banner "Backend: llama.cpp"
    [ -f "$LLAMACPP_BIN" ] || { log "[SKIP] llama-server not found: $LLAMACPP_BIN"; return; }

    for i in "${!MODEL_NAMES[@]}"; do
        local MODEL_NAME="${MODEL_NAMES[$i]}"
        local MODEL_QUANT="${MODEL_QUANTS[$i]}"
        local MODEL_PATH="${MODEL_PATHS[$i]}"
        local MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
        local MODEL_CTX_SIZE="${MODEL_CTX_SIZES[$i]}"

        [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue
        [ "$DRY_RUN" = 1 ] && { log "[DRY RUN] llamacpp: $MODEL_NAME  path=$([ -f "$MODEL_PATH" ] && echo OK || echo MISSING)"; continue; }

        local missing=0
        for G in "${GEN_LENGTHS[@]}"; do
            for P in "${PROMPT_LENGTHS[@]}"; do
                [ ! -f "$BASE_ARTIFACT/llamacpp/$MODEL_NAME/gen${G}/ctx${P}/profile_export_aiperf.json" ] && \
                    missing=$((missing + 1))
            done
        done
        if [ "$missing" = 0 ]; then
            log "  [RESUME SKIP] llamacpp/$MODEL_NAME — all combos done"
            BENCH_MODELS+=("llamacpp:$i")
            continue
        fi

        echo ""
        echo "  ┌─────────────────────────────────────────────────────┐"
        printf "  │  llamacpp / %-40s│\n" "$MODEL_NAME ($MODEL_QUANT)"
        printf "  │  ctx=%-47s│\n" "$MODEL_CTX_SIZE"
        echo "  └─────────────────────────────────────────────────────┘"

        if [ ! -f "$MODEL_PATH" ]; then
            log "  [SKIP] GGUF not found: $MODEL_PATH"
            SKIPPED_MODELS+=("llamacpp:$MODEL_NAME (file not found)")
            continue
        fi

        local SERVER_LOG="$BASE_ARTIFACT/llamacpp/${MODEL_NAME}-server.log"
        mkdir -p "$(dirname "$SERVER_LOG")"
        log "Launching llama-server on :$LLAMACPP_PORT..."
        "$LLAMACPP_BIN" -m "$MODEL_PATH" \
            --host 0.0.0.0 --port "$LLAMACPP_PORT" \
            -ngl 99 --parallel 1 -c "$MODEL_CTX_SIZE" \
            -t 1 -fa 1 --prio 2 --mlock \
            --no-cache-prompt --cache-ram 0 \
            > "$SERVER_LOG" 2>&1 &
        echo $! > "$SERVER_PIDFILE"
        log "  PID: $(cat "$SERVER_PIDFILE")"

        local READY=0 ELAPSED=0
        while [ "$ELAPSED" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
            sleep 2; ELAPSED=$((ELAPSED + 2))
            if ! kill -0 "$(cat "$SERVER_PIDFILE" 2>/dev/null)" 2>/dev/null; then
                log "  [!] Server died at t=${ELAPSED}s (OOM?)"
                grep -E "error|OOM|failed|Metal" "$SERVER_LOG" | tail -5 | sed 's/^/      /'
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
            log "  [FAIL] $MODEL_NAME — server did not start"
            SKIPPED_MODELS+=("llamacpp:$MODEL_NAME (server failed)")
            kill_llamacpp; continue
        fi

        if [ "$SKIP_SMOKE" = 1 ]; then
            log "  [SMOKE SKIP]"
        else
            if ! smoke_test "http://localhost:$LLAMACPP_PORT" "$MODEL_NAME"; then
                log "  [SMOKE FAIL] $MODEL_NAME — skipping"
                SKIPPED_MODELS+=("llamacpp:$MODEL_NAME (smoke failed)")
                kill_llamacpp; continue
            fi
            log "  [SMOKE PASS]"
        fi

        BENCH_MODELS+=("llamacpp:$i")

        local RUN_NUM=0
        local TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))
        for GEN in "${GEN_LENGTHS[@]}"; do
            for CTX in "${PROMPT_LENGTHS[@]}"; do
                RUN_NUM=$((RUN_NUM + 1))
                local ARTIFACT_DIR="$BASE_ARTIFACT/llamacpp/$MODEL_NAME/gen${GEN}/ctx${CTX}"
                mkdir -p "$ARTIFACT_DIR"
                [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [RESUME SKIP]"; continue; }
                [ "$CTX" -ge "$MODEL_CTX_SIZE" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [SKIP: prompt >= ctx_size $MODEL_CTX_SIZE]"; continue; }
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"
                if ! ensure_llamacpp_alive "$MODEL_PATH" "$MODEL_CTX_SIZE" "$SERVER_LOG"; then
                    log "  [ABORT] Cannot recover server"
                    SKIPPED_MODELS+=("llamacpp:$MODEL_NAME (server unrecoverable at gen=$GEN ctx=$CTX)")
                    stop_combo_powermetrics
                    break 2
                fi
                printf '{"model":"%s","quant":"%s","backend":"llamacpp","gen":%d,"ctx":%d}\n' \
                    "$MODEL_NAME" "$MODEL_QUANT" "$GEN" "$CTX" > "$ARTIFACT_DIR/combo_info.json"
                start_combo_powermetrics "$ARTIFACT_DIR"
                start_combo_rss "$ARTIFACT_DIR" "$(cat "$SERVER_PIDFILE")"
                # TERM=dumb prevents textual from reading mouse/special bytes that
                # cause UnicodeDecodeError in its input thread inside tmux
                TERM=dumb "$AIPERF_BIN" profile \
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
                stop_combo_rss
                stop_combo_powermetrics
                [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && { log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"; }
            done
        done

        kill_llamacpp
        log "Cooling ${COOLDOWN_MODEL}s..."
        sleep "$COOLDOWN_MODEL"
    done
}

# ══════════════════════════════════════════════════════════════════════════════
# BACKEND: Ollama
# ══════════════════════════════════════════════════════════════════════════════
run_ollama() {
    banner "Backend: Ollama"
    command -v ollama &>/dev/null || { log "[SKIP] ollama not installed"; return; }

    log "Stopping any stale llama-server..."
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null || true
    sleep 3

    log "Ensuring Ollama daemon is running..."
    if ! pgrep -f "ollama serve" &>/dev/null; then
        OLLAMA_HOST=0.0.0.0 OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=1 ollama serve &>/dev/null &
        sleep 3
    fi

    local elapsed=0
    while [ "$elapsed" -lt 30 ]; do
        local code
        code=$(curl -s "http://localhost:$OLLAMA_PORT/api/tags" --max-time 3 \
               -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        [ "$code" = "200" ] && log "  [OK] Ollama daemon ready" && break
        sleep 2; elapsed=$((elapsed + 2))
    done
    [ "$elapsed" -ge 30 ] && { log "[SKIP] Ollama daemon failed to start"; return; }

    for i in "${!MODEL_NAMES[@]}"; do
        local MODEL_NAME="${MODEL_NAMES[$i]}"
        local MODEL_QUANT="${MODEL_QUANTS[$i]}"
        local MODEL_PATH="${MODEL_PATHS[$i]}"
        local MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
        local MODEL_CTX_SIZE="${MODEL_CTX_SIZES[$i]}"

        [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue

        if [ -z "${OLLAMA_CFG[$MODEL_NAME]+x}" ]; then
            log "  [SKIP] No Ollama config for $MODEL_NAME"
            continue
        fi
        local tmpl_type="${OLLAMA_CFG[$MODEL_NAME]}"

        [ "$DRY_RUN" = 1 ] && { log "[DRY RUN] ollama: $MODEL_NAME  template=$tmpl_type"; continue; }

        local missing=0
        for G in "${GEN_LENGTHS[@]}"; do
            for P in "${PROMPT_LENGTHS[@]}"; do
                [ ! -f "$BASE_ARTIFACT/ollama/$MODEL_NAME/gen${G}/ctx${P}/profile_export_aiperf.json" ] && \
                    missing=$((missing + 1))
            done
        done
        if [ "$missing" = 0 ]; then
            log "  [RESUME SKIP] ollama/$MODEL_NAME — all combos done"
            BENCH_MODELS+=("ollama:$i")
            continue
        fi

        echo ""
        echo "  ┌─────────────────────────────────────────────────────┐"
        printf "  │  ollama / %-42s│\n" "$MODEL_NAME ($MODEL_QUANT)"
        printf "  │  template=%-42s│\n" "$tmpl_type  ctx=$MODEL_CTX_SIZE"
        echo "  └─────────────────────────────────────────────────────┘"

        local model_tag
        if ! model_tag=$(import_ollama_model \
                "$MODEL_NAME" "$MODEL_PATH" "$tmpl_type" "$MODEL_CTX_SIZE"); then
            log "  [SKIP] $MODEL_NAME — GGUF import failed"
            SKIPPED_MODELS+=("ollama:$MODEL_NAME (GGUF import failed)")
            continue
        fi

        log "  Warmup (loading $model_tag into GPU)..."
        curl -s "http://localhost:$OLLAMA_PORT/api/generate" \
            -d "{\"model\":\"$model_tag\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
            --max-time 120 -o /dev/null 2>/dev/null || true

        if [ "$SKIP_SMOKE" = 1 ]; then
            log "  [SMOKE SKIP]"
        else
            if ! smoke_test "http://localhost:$OLLAMA_PORT" "$model_tag"; then
                log "  [SMOKE FAIL] $MODEL_NAME — retrying once..."
                ollama stop "$model_tag" 2>/dev/null || true
                sleep 5
                curl -s "http://localhost:$OLLAMA_PORT/api/generate" \
                    -d "{\"model\":\"$model_tag\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
                    --max-time 120 -o /dev/null 2>/dev/null || true
                if ! smoke_test "http://localhost:$OLLAMA_PORT" "$model_tag"; then
                    log "  [SMOKE FAIL x2] $MODEL_NAME — skipping"
                    SKIPPED_MODELS+=("ollama:$MODEL_NAME (smoke failed x2)")
                    ollama stop "$model_tag" 2>/dev/null || true
                    sleep 5; continue
                fi
            fi
            log "  [SMOKE PASS]"
        fi

        BENCH_MODELS+=("ollama:$i")

        local RUN_NUM=0
        local TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))
        for GEN in "${GEN_LENGTHS[@]}"; do
            for CTX in "${PROMPT_LENGTHS[@]}"; do
                RUN_NUM=$((RUN_NUM + 1))
                local ARTIFACT_DIR="$BASE_ARTIFACT/ollama/$MODEL_NAME/gen${GEN}/ctx${CTX}"
                mkdir -p "$ARTIFACT_DIR"
                [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [RESUME SKIP]"; continue; }
                [ "$CTX" -ge "$MODEL_CTX_SIZE" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [SKIP: prompt >= ctx_size $MODEL_CTX_SIZE]"; continue; }
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"

                local code
                code=$(curl -s "http://localhost:$OLLAMA_PORT/api/tags" --max-time 5 \
                       -o /dev/null -w "%{http_code}" 2>/dev/null || true)
                if [ "$code" != "200" ]; then
                    log "  [!] Ollama not responding at gen=$GEN ctx=$CTX"
                    stop_combo_powermetrics
                    if ! restart_ollama "$model_tag"; then
                        log "  [ABORT] Ollama unrecoverable"
                        SKIPPED_MODELS+=("ollama:$MODEL_NAME (ollama died at gen=$GEN ctx=$CTX)")
                        break 2
                    fi
                    log "  [RECOVERED] Continuing from gen=$GEN ctx=$CTX"
                fi

                printf '{"model":"%s","quant":"%s","backend":"ollama","gen":%d,"ctx":%d}\n' \
                    "$MODEL_NAME" "$MODEL_QUANT" "$GEN" "$CTX" > "$ARTIFACT_DIR/combo_info.json"
                local ollama_pid
                ollama_pid=$(pgrep -f "ollama serve" | head -1 || true)
                start_combo_powermetrics "$ARTIFACT_DIR"
                start_combo_rss "$ARTIFACT_DIR" "$ollama_pid"
                TERM=dumb "$AIPERF_BIN" profile \
                    --model                         "$model_tag" \
                    --streaming \
                    --endpoint-type                 'chat' \
                    --url                           "http://localhost:$OLLAMA_PORT" \
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
                    --use-legacy-max-tokens \
                    || log "  aiperf failed (ctx=$CTX gen=$GEN)"
                stop_combo_rss
                stop_combo_powermetrics
                [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && { log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"; }
            done
        done

        log "Unloading $model_tag from GPU..."
        curl -s "http://localhost:$OLLAMA_PORT/api/generate" \
            -d "{\"model\":\"$model_tag\",\"keep_alive\":0,\"prompt\":\"\"}" \
            --max-time 10 -o /dev/null 2>/dev/null || true
        ollama stop "$model_tag" 2>/dev/null || true
        log "Cooling ${COOLDOWN_MODEL}s..."
        sleep "$COOLDOWN_MODEL"
    done
}

# ── Run backends ──────────────────────────────────────────────────────────────
banner "Running benchmarks (backend=$BACKEND)"

case "$BACKEND" in
    llamacpp) run_llamacpp ;;
    ollama)   run_ollama ;;
    both)
        run_llamacpp
        log "Backend cooldown ${COOLDOWN_BACKEND}s before Ollama..."
        sleep "$COOLDOWN_BACKEND"
        run_ollama
        ;;
esac

stop_combo_powermetrics

if [ "$DRY_RUN" = 1 ]; then
    banner "Dry run complete"; exit 0
fi
if [ "${#BENCH_MODELS[@]}" = 0 ]; then
    echo "  No models benchmarked."; exit 0
fi

# ── Generate report.md ────────────────────────────────────────────────────────
banner "Generating report.md"

cat > "$REPORT_PY" << 'PYEOF'
import re, sys, json, os, glob
from datetime import datetime

base_dir    = sys.argv[1]
skipped_arg = sys.argv[2] if len(sys.argv) > 2 else ""
report_path = os.path.join(base_dir, "report.md")
skipped     = [s for s in skipped_arg.split("||") if s] if skipped_arg else []

# ── powermetrics loader ───────────────────────────────────────────────────────
# Returns list of (epoch_s, combined_mw, cpu_c, gpu_c).
# Each block in the log starts with a "Sampled system activity" timestamp line,
# followed by power and temperature lines for that sample interval.
def load_powermetrics(log_path):
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
    return records

# ── phase-separated tok/J ─────────────────────────────────────────────────────
# Uses profile_export.jsonl (written by aiperf alongside profile_export_aiperf.json)
# for exact per-request prefill/decode timestamps, then classifies each
# powermetrics sample into the phase whose window it falls in.
#
# tok/J = OSL_p50 / (decode_power_W × p50_decode_s)
#   — decode energy only, same method as Bonsai generate_report.py
#
# Fallback: if jsonl is absent or has no profiling records, reconstructs phase
# windows from TTFT p50 and request latency p50 (less accurate but still
# better than whole-combo averaging).
def compute_tok_j(artifact_dir, pm_records):
    aiperf_path = os.path.join(artifact_dir, "profile_export_aiperf.json")
    jsonl_path  = os.path.join(artifact_dir, "profile_export.jsonl")

    try:
        d = json.load(open(aiperf_path))
    except Exception:
        return (None,) * 8

    def p50(k): return (d.get(k) or {}).get("p50")

    osl_p50     = p50("output_sequence_length")
    ttft_p50_ms = p50("time_to_first_token")
    rl_p50_ms   = p50("request_latency")
    t0_str      = d.get("start_time")

    if not pm_records:
        return (None,) * 8

    all_mw  = [mw for _, mw, _, _ in pm_records]
    all_cpu = [c  for _, _,  c, _ in pm_records if c is not None]
    all_gpu = [g  for _, _,  _, g in pm_records if g is not None]
    total_avg_pw_w = (sum(all_mw) / len(all_mw)) / 1000.0 if all_mw else None
    avg_cpu_c      = sum(all_cpu) / len(all_cpu) if all_cpu else None
    avg_gpu_c      = sum(all_gpu) / len(all_gpu) if all_gpu else None
    peak_cpu_c     = max(all_cpu) if all_cpu else None

    t0 = None
    if t0_str:
        try: t0 = datetime.fromisoformat(t0_str).timestamp()
        except Exception: pass

    # Load per-request phase windows from jsonl
    per_req = []
    if os.path.exists(jsonl_path):
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

    # Fallback: reconstruct timeline from TTFT p50 + RL p50
    if (not prefill_mw or not decode_mw) and ttft_p50_ms and rl_p50_ms and t0 is not None:
        ttft_s = ttft_p50_ms / 1000.0
        rl_s   = rl_p50_ms   / 1000.0
        n_reqs = len(per_req) or int((d.get("request_count") or {}).get("avg") or 20)
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
    decode_j     = (decode_pw_w * p50_decode_s) if (decode_pw_w and p50_decode_s) else None
    tok_j        = (osl_p50 / decode_j)         if (osl_p50 and decode_j and decode_j > 0) else None

    return tok_j, prefill_pw_w, decode_pw_w, decode_j, total_avg_pw_w, avg_cpu_c, avg_gpu_c, peak_cpu_c

# ── RSS peak ──────────────────────────────────────────────────────────────────
def parse_combo_rss(artifact_dir):
    rss_path = os.path.join(artifact_dir, "rss.log")
    samples = []
    try:
        for line in open(rss_path):
            line = line.strip()
            if line.isdigit():
                samples.append(int(line))
    except FileNotFoundError:
        return None
    return max(samples) / 1024.0 if samples else None  # KB → MB

def get_combo_quant(artifact_dir):
    try:
        return json.load(open(os.path.join(artifact_dir, "combo_info.json"))).get("quant", "?")
    except Exception:
        return "?"

# ── Collect results ───────────────────────────────────────────────────────────
results = []
for json_path in sorted(glob.glob(f"{base_dir}/**/profile_export_aiperf.json", recursive=True)):
    rel   = os.path.relpath(json_path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) < 5: continue
    backend      = parts[0]
    model_name   = parts[1]
    gen          = int(re.sub(r'\D', '', parts[2]))
    ctx          = int(re.sub(r'\D', '', parts[3]))
    artifact_dir = os.path.dirname(json_path)
    try: d = json.load(open(json_path))
    except: continue
    def g(key, stat="avg"): return (d.get(key, {}) or {}).get(stat)

    pm_records  = load_powermetrics(os.path.join(artifact_dir, "powermetrics.log"))
    tok_j, prefill_pw, decode_pw, decode_j, total_avg_pw, avg_cpu_c, avg_gpu_c, peak_cpu_c = \
        compute_tok_j(artifact_dir, pm_records)
    peak_rss_mb = parse_combo_rss(artifact_dir)

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
        "tps": g("output_token_throughput_per_user"),
        "tps_p50": g("output_token_throughput_per_user","p50"),
        "req_s": g("request_throughput"),
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
        # power
        "total_avg_pw": total_avg_pw,
        "prefill_pw":   prefill_pw,
        "decode_pw":    decode_pw,
        "decode_j":     decode_j,
        "avg_cpu_c":    avg_cpu_c,
        "avg_gpu_c":    avg_gpu_c,
        "peak_cpu_c":   peak_cpu_c,
        "tok_j":        tok_j,        # OSL_p50 / decode_j  (decode-energy only)
        "peak_rss_mb":  peak_rss_mb,
    })
results.sort(key=lambda r: (r["backend"], r["model"], r["gen"], r["prompt"]))

# ── Thermal summary (per backend:model, across all combos) ───────────────────
thermal_acc = {}
for r in results:
    k = f"{r['backend']}:{r['model']}"
    thermal_acc.setdefault(k, {"pw": [], "cpu": [], "gpu": [], "tj": []})
    if r["total_avg_pw"] is not None: thermal_acc[k]["pw"].append(r["total_avg_pw"])
    if r["avg_cpu_c"]    is not None: thermal_acc[k]["cpu"].append(r["avg_cpu_c"])
    if r["avg_gpu_c"]    is not None: thermal_acc[k]["gpu"].append(r["avg_gpu_c"])
    if r["peak_cpu_c"]   is not None: thermal_acc[k]["tj"].append(r["peak_cpu_c"])

thermal = {}
for k, v in thermal_acc.items():
    peak = max(v["tj"]) if v["tj"] else None
    thermal[k] = {
        "avg_pw":   sum(v["pw"])  / len(v["pw"])  if v["pw"]  else None,
        "avg_cpu":  sum(v["cpu"]) / len(v["cpu"]) if v["cpu"] else None,
        "avg_gpu":  sum(v["gpu"]) / len(v["gpu"]) if v["gpu"] else None,
        "peak_cpu": peak,
        "throttled": peak is not None and peak > 95,
    }

best_tokj = {}
for r in results:
    k = f"{r['backend']}:{r['model']}"
    if r["tok_j"] is not None:
        if k not in best_tokj or r["tok_j"] > best_tokj[k]["tok_j"]: best_tokj[k] = r

lines = []
def L(s=""): lines.append(s)
def fmt(v, fs, fb="—"): return format(v, fs) if v is not None else fb

L("# LLM Benchmark — Mac Mini M4 16GB")
L()
L(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
L(f"**Backends:** {', '.join(sorted(set(r['backend'] for r in results)))}  ")
L(f"**Sweep:** prompt ∈ {{256,512,1024,2048,4096}} tok × gen ∈ {{256,512,1024}} tok  ")
L(f"**Power:** macOS powermetrics — Combined (CPU + GPU + ANE) at 500 ms  ")
L(f"**Tok/J:** OSL p50 / (decode power W × p50 decode s) — decode phase only  ")
L(f"**Artifacts:** `{base_dir}`")
L()

if skipped:
    L("## Skipped / Failed Models")
    for s in skipped: L(f"- {s}")
    L()

for backend in sorted(set(r["backend"] for r in results)):
    backend_results = [r for r in results if r["backend"] == backend]
    L(f"## Full Results — {backend}")
    L()
    L("> **Prefill W** / **Decode W**: median powermetrics samples classified by per-request phase windows from `profile_export.jsonl`.  ")
    L("> **Tok/J** = OSL p50 ÷ (Decode W × p50 decode duration) — decode energy only.")
    L()
    H = ("| Model | Quant | ISL | OSL | OSL mis% "
         "| TTFT p50 | p90 | p99 "
         "| ITL p50 | p90 | p99 "
         "| Tok/s p50 | Req/s "
         "| RL p50 | p90 | p99 "
         "| Prefill TPS p50 | p90 "
         "| Total W | Prefill W | Decode W | Peak RAM MB | **Tok/J** |")
    S = ("|-------|:-----:|---:|---:|---:"
         "|---:|---:|---:"
         "|---:|---:|---:"
         "|---:|---:"
         "|---:|---:|---:"
         "|---:|---:"
         "|---:|---:|---:|---:|---:|")
    L(H); L(S)
    for r in backend_results:
        L(f"| {r['model']} | {r['quant']} "
          f"| {fmt(r['isl'],'.0f')} | {fmt(r['osl'],'.1f')} | {fmt(r['osl_mis'],'.2f')} "
          f"| {fmt(r['ttft_p50'],'.1f')} | {fmt(r['ttft_p90'],'.1f')} | {fmt(r['ttft_p99'],'.1f')} "
          f"| {fmt(r['itl_p50'],'.2f')} | {fmt(r['itl_p90'],'.2f')} | {fmt(r['itl_p99'],'.2f')} "
          f"| {fmt(r['tps_p50'],'.2f')} | {fmt(r['req_s'],'.3f')} "
          f"| {fmt(r['rl_p50'],'.1f')} | {fmt(r['rl_p90'],'.1f')} | {fmt(r['rl_p99'],'.1f')} "
          f"| {fmt(r['pre_p50'],'.1f')} | {fmt(r['pre_p90'],'.1f')} "
          f"| {fmt(r['total_avg_pw'],'.2f')} | {fmt(r['prefill_pw'],'.2f')} | {fmt(r['decode_pw'],'.2f')} "
          f"| {fmt(r['peak_rss_mb'],'.0f')} | **{fmt(r['tok_j'],'.4f')}** |")
    L()

L("## Best Tok/J per Model")
L()
L("| Backend | Model | Quant | **Best Tok/J** | ISL | OSL | Tok/s p50 | Prefill W | Decode W |")
L("|---------|-------|:-----:|--------------:|---:|---:|---:|---:|---:|")
for key in sorted(best_tokj.keys()):
    b = best_tokj[key]
    L(f"| {b['backend']} | {b['model']} | {b['quant']} | **{fmt(b['tok_j'],'.4f')}** "
      f"| {fmt(b['isl'],'.0f')} | {fmt(b['osl'],'.1f')} "
      f"| {fmt(b['tps_p50'],'.2f')} | {fmt(b['prefill_pw'],'.2f')} | {fmt(b['decode_pw'],'.2f')} |")

L()
L("## Thermal Summary")
L()
L("| Backend | Model | Avg Total W | Avg CPU °C | Avg GPU °C | Peak CPU °C | Throttled |")
L("|---------|-------|---:|---:|---:|---:|:---:|")
for key in sorted(thermal.keys()):
    t = thermal[key]
    backend, name = key.split(":", 1)
    L(f"| {backend} | {name} | {fmt(t['avg_pw'],'.2f')} | {fmt(t['avg_cpu'],'.1f')} "
      f"| {fmt(t['avg_gpu'],'.1f')} | {fmt(t['peak_cpu'],'.1f')} "
      f"| {'YES ⚠' if t['throttled'] else 'No'} |")

L()
L("## Methodology")
L()
L("- **Power rail**: Combined (CPU + GPU + ANE) from macOS `powermetrics` at 50 ms interval")
L("- **Phase separation**: exact per-request prefill/decode windows from `profile_export.jsonl`")
L("  (`request_start_ns` → `request_ack_ns` = prefill, `request_ack_ns` → `request_end_ns` = decode)")
L("- **Prefill / Decode W**: median of powermetrics samples classified into each phase")
L("- **Tok/J**: `OSL_p50 / (decode_power_W × p50_decode_s)` — output tokens per joule of decode energy")
L("- **Fallback**: timeline reconstruction from TTFT p50 + RL p50 if `profile_export.jsonl` absent")
L("- **Peak RAM**: max RSS of inference process sampled at 500 ms during combo run")
L("- **Concurrency**: 1")
L()
L("---")
L(f"*Generated by `benchmark_non_reasoning.sh` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
L(f"*{len(results)} rows  |  {len(set(r['backend'] for r in results))} backends  |  {len(set(r['model'] for r in results))} models*")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  Report → {report_path}")
print(f"  {len(results)} rows across {len(sorted(set(r['backend'] for r in results)))} backend(s)")
PYEOF

SKIPPED_STR=""
for s in "${SKIPPED_MODELS[@]}"; do SKIPPED_STR="${SKIPPED_STR}${s}||"; done

python3 "$REPORT_PY" "$BASE_ARTIFACT" "$SKIPPED_STR"

banner "Done"
echo "  Artifacts : $BASE_ARTIFACT"
echo "  Report    : $BASE_ARTIFACT/report.md"
echo ""

deactivate 2>/dev/null || true
