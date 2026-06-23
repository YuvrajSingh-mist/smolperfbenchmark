#!/bin/bash
# bench-non-reasoning.sh — Blog benchmark: tiny LLMs on Raspberry Pi 5
#
# Per model: ONE server launch → smoke → aiperf sweep.
# Sweeps: prompt in {128,512,1024,2048} × gen in {64,128,256}
# Key metric: output tok/J (output tokens per joule), from aiperf + PMIC power log.
#
# Usage:
#   bash bench-non-reasoning.sh                          # llamacpp only, all models
#   bash bench-non-reasoning.sh --backend ollama         # ollama only
#   bash bench-non-reasoning.sh --backend both           # llamacpp then ollama
#   bash bench-non-reasoning.sh --reqs 5
#   bash bench-non-reasoning.sh --only smollm2
#   bash bench-non-reasoning.sh --skip-smoke
#   bash bench-non-reasoning.sh --dry-run
#   bash bench-non-reasoning.sh --resume DIR

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
    
    # Try system package first (Debian 13 Trixie / PEP 668 compatible)
    if command -v apt-get &>/dev/null; then
        echo "Attempting system package install..."
        sudo apt-get update -qq && sudo apt-get install -y python3-huggingface-hub 2>/dev/null || true
         
        # Add ~/.local/bin to PATH if needed
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # If still not found, use pip with appropriate flags
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
        # Debian 13 Trixie requires --break-system-packages for global install
        $PIP_CMD install --break-system-packages -U "huggingface_hub[cli]" || \
        $PIP_CMD install --user -U "huggingface_hub[cli]"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Final verification
    if ! command -v hf &>/dev/null; then
        echo "ERROR: hf command still not available after install attempts."
        echo "Try manually: pip3 install --break-system-packages 'huggingface_hub[cli]'"
        exit 1
    fi
    echo "Hugging Face CLI installed successfully."
fi

# ── Ensure Ollama v0.24.0 is installed ───────────────────────────────────────
# v0.24.0 is required: it's the version that supports local GGUF import for all
# benchmark models (gemma3-4b, lfm2.5-350m, lfm2.5-1.2b). Later versions
# (e.g. 0.30.x) introduced a llama-quantize validation regression that rejects
# these GGUFs with "compatibility patches" errors.
REQUIRED_OLLAMA_VERSION="0.24.0"
_current_ollama_ver=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ "${_current_ollama_ver}" != "${REQUIRED_OLLAMA_VERSION}" ]; then
    echo "Ollama ${_current_ollama_ver:-not found} — installing v${REQUIRED_OLLAMA_VERSION}..."
    curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="${REQUIRED_OLLAMA_VERSION}" sudo -E sh
    if [ "$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" != "${REQUIRED_OLLAMA_VERSION}" ]; then
        echo "ERROR: Failed to install Ollama v${REQUIRED_OLLAMA_VERSION}. Aborting."
        exit 1
    fi
    echo "Ollama v${REQUIRED_OLLAMA_VERSION} installed successfully."
fi
unset _current_ollama_ver

# ── Auto-relaunch inside tmux if not already there ────────────────────────────
if [ -z "${TMUX:-}" ]; then
    SESSION="non-reasoning-bench"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "bash $(realpath "$0") $([ $# -gt 0 ] && printf '%q ' "$@"); echo 'Done — press Enter to exit'; read"
    echo "Launched in tmux session '$SESSION'"
    echo "Attach with:  tmux attach -t $SESSION"
    exit 0
fi

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND="llamacpp"    # llamacpp | ollama | both
REQS=20
ONLY_MODEL=""
ONLY_COMBO=""   # comma-separated GEN:CTX pairs, e.g. "256:1024" or "256:1024,64:512"
SKIP_SMOKE=0
DRY_RUN=0
RESUME_DIR=""
CONCURRENCY=1
SLICE_DURATION=30
RANDOM_SEED=42
REQUEST_TIMEOUT=180
COOLDOWN_COMBO=10
COOLDOWN_MODEL=30
COOLDOWN_BACKEND=45   # wait between llamacpp and ollama runs
SERVER_STARTUP_TIMEOUT=300

PROMPT_LENGTHS=(128 512 1024 2048)
GEN_LENGTHS=(64 128 256)
CONTEXT_SIZE=2560   # max_prompt(2048) + max_gen(256) = 2304, padded to 2560

LLAMACPP_BIN="${LLAMACPP_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
LLAMACPP_PORT=8080
OLLAMA_PORT=11434

# Thread env vars for llama.cpp are set inline at llama-server launch only.
# Do NOT export them globally — Ollama must not inherit them.

GGUF_DIR="$HOME/gguf-models"
TEGRA_PIDFILE="/tmp/blog_bench_tegrastats.pid"
SERVER_PIDFILE="/tmp/blog_bench_server.pid"
REPORT_PY="/tmp/blog_report.py"

# ── Model table: name|quant|gguf_path|tokenizer|ctx_size ─────────────────────
declare -a MODELS=(
    "smollm2-135m|Q4_K_M|$GGUF_DIR/smollm2-135mq4_k_m.gguf|HuggingFaceTB/SmolLM2-135M-Instruct|2560"
    "smollm2-360m|Q8_0|$GGUF_DIR/smollm2-360mq8_0|HuggingFaceTB/SmolLM2-360M-Instruct|2560"
    "qwen2.5-0.5b|Q4_K_M|$GGUF_DIR/qwen2-5-0-5bq4_k_m|Qwen/Qwen2.5-0.5B-Instruct|2560"
    "qwen3-0.6b|Q8_0|$GGUF_DIR/qwen3-0-6bq8_0|Qwen/Qwen3-0.6B|2560"
    "llama3.2-1b|Q4_K_M|$GGUF_DIR/llama3-2-1bq4_k_m.gguf|meta-llama/Llama-3.2-1B-Instruct|2560"
    "gemma3-1b|Q4_K_M|$GGUF_DIR/gemma3-1b-q4_k_m.gguf|google/gemma-3-1b-it|2560"
    "gemma3-4b|Q4_K_M|$GGUF_DIR/gemma3-4b-q4_k_m.gguf|google/gemma-3-4b-it|2560"
    "lfm2.5-350m|Q4_K_M|$GGUF_DIR/lfm2.5-350m-q4_k_m.gguf|LiquidAI/LFM2.5-350M|2560"
    "lfm2.5-1.2b|Q4_K_M|$GGUF_DIR/lfm2.5-1.2b-q4_k_m.gguf|LiquidAI/LFM2.5-1.2B-Instruct|2560"
)

# ── GGUF download sources: local_filename -> "hf_repo hf_filename" ────────────
declare -A GGUF_SOURCES=(
    ["smollm2-135mq4_k_m.gguf"]="bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf"
    ["smollm2-360mq8_0"]="bartowski/SmolLM2-360M-Instruct-GGUF SmolLM2-360M-Instruct-Q8_0.gguf"
    ["qwen2-5-0-5bq4_k_m"]="Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ["qwen3-0-6bq8_0"]="Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf"
    ["llama3-2-1bq4_k_m.gguf"]="bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ["gemma3-1b-q4_k_m.gguf"]="lmstudio-community/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf"
    ["gemma3-4b-q4_k_m.gguf"]="lmstudio-community/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf"
    ["lfm2.5-350m-q4_k_m.gguf"]="LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf"
    ["lfm2.5-1.2b-q4_k_m.gguf"]="LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
)

# ── Ollama per-model config: name -> "template_type" ─────────────────────────
# No registry fallback — GGUF must be present. If import fails, model is skipped.
# template_type: chatml | chatml-smollm2 | lfm | llama3 | gemma
# Stop tokens are set per template_type inside make_ollama_modelfile().
declare -A OLLAMA_CFG=(
    # SmolLM2 uses <|im_start|> as BOS token — needs both as stop tokens.
    # Source: HuggingFaceTB/SmolLM2-135M-Instruct tokenizer_config.json
    ["smollm2-135m"]="chatml-smollm2"
    ["smollm2-360m"]="chatml-smollm2"

    # Qwen2.5: standard ChatML, EOS=<|im_end|>, no BOS.
    # Source: Qwen/Qwen2.5-0.5B-Instruct tokenizer_config.json
    ["qwen2.5-0.5b"]="chatml"

    # Qwen3: standard ChatML, EOS=<|im_end|>, no BOS.
    # Source: Qwen/Qwen3-0.6B tokenizer_config.json
    ["qwen3-0.6b"]="chatml"

    # LFM2.5: ChatML with <|startoftext|> BOS, stop on <|im_end|> and <|endoftext|>.
    # Source: LiquidAI/LFM2.5-350M tokenizer_config.json + docs.liquid.ai chat template
    ["lfm2.5-350m"]="lfm"
    ["lfm2.5-1.2b"]="lfm"

    # Llama-3.2: uses header-based format. BOS=<|begin_of_text|> (added by tokenizer).
    # Stop: <|eot_id|>, <|start_header_id|>, <|end_header_id|>.
    # Source: meta-llama/Llama-3.2-1B-Instruct tokenizer_config.json + ollama.com/library/llama3.2
    ["llama3.2-1b"]="llama3"

    # Gemma3: NO system role support (raises exception in official template).
    # System messages merged into the user turn. BOS token=<bos> (added by tokenizer).
    # Stop: <end_of_turn>, <start_of_turn>. Role name is "model" not "assistant".
    # Source: google/gemma-3-1b-it tokenizer_config.json + ai.google.dev/gemma/docs
    ["gemma3-1b"]="gemma"
    ["gemma3-4b"]="gemma"
)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)     BACKEND="$2";      shift 2 ;;
        --reqs)        REQS="$2";         shift 2 ;;
        --only)        ONLY_MODEL="$2";   shift 2 ;;
        --only-combo)  ONLY_COMBO="$2";   shift 2 ;;
        --skip-smoke)  SKIP_SMOKE=1;      shift ;;
        --dry-run)     DRY_RUN=1;         shift ;;
        --resume)      RESUME_DIR="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ "$BACKEND" =~ ^(llamacpp|ollama|both)$ ]] || { echo "ERROR: --backend must be llamacpp|ollama|both"; exit 1; }

# ── Resolve artifact dir ──────────────────────────────────────────────────────
if [ -n "$RESUME_DIR" ]; then
    [ ! -d "$RESUME_DIR" ] && echo "ERROR: --resume dir not found: $RESUME_DIR" && exit 1
    BASE_ARTIFACT="$RESUME_DIR"
    echo "  [RESUME] Reusing artifact dir: $BASE_ARTIFACT"
else
    BASE_ARTIFACT="$HOME/Desktop/smolbenchmark/benchmark-raspberrypi5/artifacts/blog-all-$(date +%Y%m%d-%H%M)"
fi


TIMING_LOG="$BASE_ARTIFACT/model_timing.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
banner() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
log()    { echo "  [$(date +%H:%M:%S)] $*"; }

stop_stats_logger() {
    [ -f "$TEGRA_PIDFILE" ] && kill "$(cat "$TEGRA_PIDFILE")" 2>/dev/null || true
    rm -f "$TEGRA_PIDFILE"
}

kill_llamacpp() {
    if [ -f "$SERVER_PIDFILE" ]; then
        kill "$(cat "$SERVER_PIDFILE")" 2>/dev/null || true
        rm -f "$SERVER_PIDFILE"
    fi
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null || true
    sleep 3
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
    BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LLAMA_ARG_THREADS=4 \
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

# ── Smoke test (shared between backends) ──────────────────────────────────────
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

# ── Ollama max_tokens sanity check ───────────────────────────────────────────
# Ollama requires the legacy 'max_tokens' field; newer aiperf defaults to
# 'max_completion_tokens' which Ollama ignores, causing unconstrained generation
# and ~90% request timeouts. This runs once per ollama backend invocation.
check_ollama_max_tokens() {
    local url="$1" model_tag="$2"
    log "  ── max_tokens sanity check ──────────────────────────────"
    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'model': sys.argv[1],
    'messages': [{'role': 'user', 'content': 'Count from 1 upward, one number per line.'}],
    'max_tokens': 8,
    'stream': False
}))
" "$model_tag")
    local resp
    resp=$(curl -s "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload" --max-time 30 2>/dev/null)
    local result
    result=$(python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    ct = (d.get('usage') or {}).get('completion_tokens', -1)
    if ct < 0:
        content = (d.get('choices') or [{}])[0].get('message', {}).get('content', '')
        ct = max(1, len(content) // 4)
    print('FAIL:' + str(ct) if ct > 16 else 'OK:' + str(ct))
except Exception as e:
    print('ERR:' + str(e))
" <<< "$resp")
    if [[ "$result" == OK:* ]]; then
        log "  [OK] Ollama respects max_tokens (${result#OK:} completion tokens for limit=8)"
        log "  ──────────────────────────────────────────────────────"
        return 0
    else
        log "  ──────────────────────────────────────────────────────"
        log ""
        log "  ╔══════════════════════════════════════════════════════╗"
        log "  ║  ABORT: Ollama is ignoring max_tokens                ║"
        log "  ╠══════════════════════════════════════════════════════╣"
        log "  ║  Result: $result"
        log "  ║  Without max_tokens enforcement, the model generates ║"
        log "  ║  until EOS — aiperf requests will timeout and all    ║"
        log "  ║  benchmark data will be based on 1-5 samples.        ║"
        log "  ║                                                      ║"
        log "  ║  Check: ollama --version  (need a recent build)      ║"
        log "  ╚══════════════════════════════════════════════════════╝"
        log ""
        return 1
    fi
}

# ── Ollama: build Modelfile content ──────────────────────────────────────────
# Args: $1=gguf_path  $2=template_type  $3=ctx_size
make_ollama_modelfile() {
    local gguf_path="$1" tmpl_type="$2" ctx_size="$3"

    # All templates share the same PARAMETER block at the top.
    # num_ctx: Ollama's context window. num_keep=-1: keep all tokens in context.
    # Print the FROM line and shared PARAMETERs first — every Modelfile must start with FROM.
    printf 'FROM %s\nPARAMETER num_ctx %s\nPARAMETER num_keep -1\n' \
        "$gguf_path" "$ctx_size"

    case "$tmpl_type" in

        # ── ChatML (Qwen2.5, Qwen3) ──────────────────────────────────────────
        # EOS=<|im_end|> only. No BOS token in these models.
        # Source: Qwen tokenizer_config.json (bos_token: null)
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

        # ── ChatML SmolLM2 ────────────────────────────────────────────────────
        # BOS token is <|im_start|> (token_id=0 acts as BOS in SmolLM2 vocab).
        # Stop on both <|im_end|> and <|im_start|> to prevent run-on generation.
        # Source: HuggingFaceTB/SmolLM2-135M-Instruct tokenizer_config.json
        #   bos_token: "<|im_start|>", eos_token: "<|im_end|>"
        chatml-smollm2)
            cat <<'TMPL'
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
TMPL
            ;;

        # ── LFM2.5 ────────────────────────────────────────────────────────────
        # BOS=<|startoftext|> must be prepended explicitly (added_tokens_encoder).
        # EOS=<|im_end|>, also stop on <|endoftext|> which appears at sequence end.
        # Source: LiquidAI/LFM2.5-350M tokenizer_config.json + docs.liquid.ai
        lfm)
            cat <<'TMPL'
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
TEMPLATE """<|startoftext|>{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
TMPL
            ;;

        # ── Llama 3.2 ─────────────────────────────────────────────────────────
        # BOS=<|begin_of_text|> is added automatically by the Llama tokenizer.
        # Stop on all three special tokens to prevent partial header generation.
        # Source: meta-llama/Llama-3.2-1B-Instruct tokenizer_config.json
        #   added_tokens: <|eot_id|>=128001, <|start_header_id|>=128006, <|end_header_id|>=128007
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
        # Official template RAISES an exception for system role — unsupported.
        # We merge .System into the user turn if provided (consistent with HF behavior).
        # BOS=<bos> (token_id=2) added automatically by the Gemma tokenizer.
        # Role name is "model" (not "assistant") — this is critical.
        # Source: google/gemma-3-1b-it tokenizer_config.json + ai.google.dev/gemma/docs
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

# ── Ollama: import GGUF and return the tag to use ─────────────────────────────
# Prints the model tag to stdout. Returns 1 if both import and registry fail.
import_ollama_model() {
    local model_name="$1" gguf_path="$2" tmpl_type="$3" ctx_size="$4"
    local import_tag="local-${model_name}"

    # All log() calls in this function MUST go to stderr — this function is called
    # via $() command substitution, so stdout is captured as the return value (model tag).
    # Any log output on stdout would corrupt the model_tag variable.
    if [ ! -f "$gguf_path" ]; then
        log "  [FAIL] GGUF not found: $gguf_path" >&2
        return 1
    fi

    log "  Importing GGUF as $import_tag…" >&2
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
banner "Blog LLM Benchmark  |  Raspberry Pi 5"
echo "  Date      : $(date --iso-8601=seconds)"
echo "  Backend   : $BACKEND"
echo "  Requests  : $REQS per run"
echo "  Prompts   : ${PROMPT_LENGTHS[*]}"
echo "  Gen lens  : ${GEN_LENGTHS[*]}"
echo "  Artifacts : $BASE_ARTIFACT"
[ -n "$ONLY_MODEL" ] && echo "  Filter    : $ONLY_MODEL"
[ -n "$ONLY_COMBO" ] && echo "  Combo     : $ONLY_COMBO"
[ "$DRY_RUN"   = 1 ] && echo "  Mode      : DRY RUN"
[ -n "$RESUME_DIR" ] && echo "  Mode      : RESUME"

mkdir -p "$BASE_ARTIFACT"

# ── Initial cleanup ───────────────────────────────────────────────────────────
banner "Cleanup"
pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null && log "killed llama-server" || true
stop_stats_logger
rm -f "$SERVER_PIDFILE"

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
    tmp_path="$GGUF_DIR/${hf_file}"
    if hf download "$hf_repo" "$hf_file" --local-dir "$GGUF_DIR"; then
        [ -f "$tmp_path" ] && mv "$tmp_path" "$local_path" && log "  [DONE] $local_name"
    else
        log "  [FAIL] Could not download $local_name — model will be skipped"
    fi
done

# ── Pi 5 clock info ───────────────────────────────────────────────────────────
if [ "$DRY_RUN" = 0 ]; then
    banner "Clock info  (Raspberry Pi 5)"
    arm_hz=$(vcgencmd measure_clock arm 2>/dev/null | cut -d= -f2 || echo "?")
    log "ARM clock : $(awk "BEGIN{printf \"%.0f\", ${arm_hz}/1e6}")MHz"
    log "Temp      : $(vcgencmd measure_temp 2>/dev/null || echo '?')"
    log "Throttled : $(vcgencmd get_throttled 2>/dev/null || echo '?')"
fi

# ── Activate aiperf venv ──────────────────────────────────────────────────────
source "$HOME/venv/bin/activate" 2>/dev/null || \
source "$HOME/aiperf-env/bin/activate" 2>/dev/null || {
    echo ""
    echo "  ╔══════════════════════════════════════════════════════╗"
    echo "  ║  ERROR: aiperf venv not found                        ║"
    echo "  ╠══════════════════════════════════════════════════════╣"
    echo "  ║  Looked for: ~/venv  and  ~/aiperf-env               ║"
    echo "  ║                                                      ║"
    echo "  ║  Create it with:                                     ║"
    echo "  ║    python3 -m venv ~/aiperf-env                      ║"
    echo "  ║    source ~/aiperf-env/bin/activate                  ║"
    echo "  ║    pip install aiperf                                ║"
    echo "  ╚══════════════════════════════════════════════════════╝"
    echo ""
    exit 1
}

# PMIC stats logger runs per-combo (started just before each aiperf call)
start_stats_logger() {
    local outfile="$1"
    stop_stats_logger
    (while true; do
        ts=$(date '+%m-%d-%Y %H:%M:%S')
        pmic_mw=$(vcgencmd pmic_read_adc 2>/dev/null | awk '
            /current/ { name=$1; sub(/_A$/,"",name); split($2,p,"="); curr[name]=p[2]+0 }
            /volt/    { name=$1; sub(/_V$/,"",name); split($2,p,"="); volt[name]=p[2]+0 }
            END {
                total=0
                for (r in curr) if (r in volt) total += curr[r]*volt[r]
                printf "%d", total * 1000
            }' || echo 0)
        temp_mc=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
        temp_c=$(awk "BEGIN{printf \"%.1f\", $temp_mc/1000}")
        printf '%s VDD_CPU_GPU_CV %smW cpu@%sC gpu@%sC tj@%sC\n' \
            "$ts" "$pmic_mw" "$temp_c" "$temp_c" "$temp_c" >> "$outfile"
        sleep 0.5
    done) &
    echo $! > "$TEGRA_PIDFILE"
}

declare -a SKIPPED_MODELS=() BENCH_MODELS=()

# ══════════════════════════════════════════════════════════════════════════════
# BACKEND: llama.cpp
# ══════════════════════════════════════════════════════════════════════════════
run_llamacpp() {
    banner "Backend: llama.cpp"
    if [ ! -f "$LLAMACPP_BIN" ]; then
        echo ""
        echo "  ╔══════════════════════════════════════════════════════╗"
        echo "  ║  ERROR: llama-server binary not found                ║"
        echo "  ╠══════════════════════════════════════════════════════╣"
        printf "  ║  Expected: %-42s║\n" "$LLAMACPP_BIN"
        echo "  ║                                                      ║"
        echo "  ║  Build it first:                                     ║"
        echo "  ║    cd ~/llama.cpp                                    ║"
        echo "  ║    cmake -B build -DGGML_NATIVE=ON                   ║"
        echo "  ║    cmake --build build -j\$(nproc) -t llama-server    ║"
        echo "  ║                                                      ║"
        echo "  ║  Or point to an existing binary:                     ║"
        echo "  ║    LLAMACPP_BIN=/path/to/llama-server bash $0  ║"
        echo "  ╚══════════════════════════════════════════════════════╝"
        echo ""
        exit 1
    fi

    for i in "${!MODEL_NAMES[@]}"; do
        local MODEL_NAME="${MODEL_NAMES[$i]}"
        local MODEL_QUANT="${MODEL_QUANTS[$i]}"
        local MODEL_PATH="${MODEL_PATHS[$i]}"
        local MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
        local MODEL_CTX_SIZE="${MODEL_CTX_SIZES[$i]}"

        [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue
        [ "$DRY_RUN" = 1 ] && { log "[DRY RUN] llamacpp: $MODEL_NAME  path=$([ -f "$MODEL_PATH" ] && echo OK || echo MISSING)"; continue; }

        # Resume: skip if all combos already done
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
        BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LLAMA_ARG_THREADS=4 \
        "$LLAMACPP_BIN" -m "$MODEL_PATH" \
            --host 0.0.0.0 --port "$LLAMACPP_PORT" \
            -ngl 99 --parallel 1 -c "$MODEL_CTX_SIZE" \
            --no-cache-prompt --cache-ram 0 \
            > "$SERVER_LOG" 2>&1 &
        echo $! > "$SERVER_PIDFILE"
        log "  PID: $(cat "$SERVER_PIDFILE")"

        local READY=0 ELAPSED=0
        while [ "$ELAPSED" -lt "$SERVER_STARTUP_TIMEOUT" ]; do
            sleep 2; ELAPSED=$((ELAPSED + 2))
            if ! kill -0 "$(cat "$SERVER_PIDFILE" 2>/dev/null)" 2>/dev/null; then
                log "  [!] Server died at t=${ELAPSED}s (OOM?)"
                grep -E "error|OOM|failed|CUDA" "$SERVER_LOG" | tail -5 | sed 's/^/      /'
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
        echo "MODEL_START:llamacpp:${MODEL_NAME}:${MODEL_QUANT}:$(date +%s)" >> "$TIMING_LOG"

        local RUN_NUM=0
        local TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))
        for GEN in "${GEN_LENGTHS[@]}"; do
            for CTX in "${PROMPT_LENGTHS[@]}"; do
                RUN_NUM=$((RUN_NUM + 1))
                local ARTIFACT_DIR="$BASE_ARTIFACT/llamacpp/$MODEL_NAME/gen${GEN}/ctx${CTX}"
                mkdir -p "$ARTIFACT_DIR"
                [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [RESUME SKIP]"; continue; }
                [[ -n "$ONLY_COMBO" && ",$ONLY_COMBO," != *",${GEN}:${CTX},"* ]] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [COMBO SKIP]"; continue; }
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"
                if ! ensure_llamacpp_alive "$MODEL_PATH" "$MODEL_CTX_SIZE" "$SERVER_LOG"; then
                    log "  [ABORT] Cannot recover server"
                    SKIPPED_MODELS+=("llamacpp:$MODEL_NAME (server unrecoverable at gen=$GEN ctx=$CTX)")
                    break 2
                fi
                [ "$DRY_RUN" = 0 ] && start_stats_logger "$ARTIFACT_DIR/power.log"
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
                [ "$DRY_RUN" = 0 ] && stop_stats_logger
                [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && { log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"; }
            done
        done

        echo "MODEL_END:llamacpp:${MODEL_NAME}:${MODEL_QUANT}:$(date +%s)" >> "$TIMING_LOG"
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

    # Stop any stale llama-server before starting Ollama
    log "Stopping stale llama-server processes..."
    pkill -f "llama-server.*$LLAMACPP_PORT" 2>/dev/null || true
    sleep 5

    # Restart Ollama with benchmark-tuned settings.
    # OLLAMA_NUM_PARALLEL=1 is critical: without it, aiperf fires all requests
    # concurrently and each gets a fraction of the CPU, collapsing tok/s to ~0.5.
    # We stop the system service (if any) and start our own instance so these vars
    # take effect regardless of whether systemd or a manual `ollama serve` is used.
    log "Restarting Ollama with benchmark settings (NUM_PARALLEL=1)..."
    sudo systemctl stop ollama 2>/dev/null || true
    pkill -f 'ollama serve' 2>/dev/null || true
    sleep 3
    OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_MAX_LOADED_MODELS=1 \
        ollama serve &>/dev/null &
    OLLAMA_BENCH_PID=$!

    local elapsed=0
    while [ "$elapsed" -lt 30 ]; do
        local code
        code=$(curl -s "http://localhost:$OLLAMA_PORT/api/tags" --max-time 3 \
               -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        [ "$code" = "200" ] && log "  [OK] Ollama daemon ready" && break
        sleep 2; elapsed=$((elapsed + 2))
    done
    [ "$elapsed" -ge 30 ] && { log "[SKIP] Ollama daemon failed to start"; return; }

    local _max_tokens_checked=0

    for i in "${!MODEL_NAMES[@]}"; do
        local MODEL_NAME="${MODEL_NAMES[$i]}"
        local MODEL_QUANT="${MODEL_QUANTS[$i]}"
        local MODEL_PATH="${MODEL_PATHS[$i]}"
        local MODEL_TOKENIZER="${MODEL_TOKENIZERS[$i]}"
        local MODEL_CTX_SIZE="${MODEL_CTX_SIZES[$i]}"

        [[ -n "$ONLY_MODEL" && "${MODEL_NAME,,}" != *"${ONLY_MODEL,,}"* ]] && continue

        # Look up Ollama config for this model
        if [ -z "${OLLAMA_CFG[$MODEL_NAME]+x}" ]; then
            log "  [SKIP] No Ollama config for $MODEL_NAME"
            continue
        fi
        local tmpl_type="${OLLAMA_CFG[$MODEL_NAME]}"

        [ "$DRY_RUN" = 1 ] && { log "[DRY RUN] ollama: $MODEL_NAME  template=$tmpl_type"; continue; }

        # Resume: skip if all combos already done
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

        # Import GGUF into Ollama — no fallback, error out if GGUF missing or import fails
        local model_tag
        if ! model_tag=$(import_ollama_model \
                "$MODEL_NAME" "$MODEL_PATH" "$tmpl_type" "$MODEL_CTX_SIZE"); then
            log "  [SKIP] $MODEL_NAME — GGUF import failed (see above)"
            SKIPPED_MODELS+=("ollama:$MODEL_NAME (GGUF import failed)")
            continue
        fi

        # Warm up: load model into CPU memory
        log "  Warmup (loading $model_tag into CPU)..."
        curl -s "http://localhost:$OLLAMA_PORT/api/generate" \
            -d "{\"model\":\"$model_tag\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
            --max-time 120 -o /dev/null 2>/dev/null || true

        # One-time check that Ollama respects max_tokens (first model only)
        if [ "$_max_tokens_checked" = 0 ]; then
            if ! check_ollama_max_tokens "http://localhost:$OLLAMA_PORT" "$model_tag"; then
                exit 1
            fi
            _max_tokens_checked=1
        fi

        if [ "$SKIP_SMOKE" = 1 ]; then
            log "  [SMOKE SKIP]"
        else
            if ! smoke_test "http://localhost:$OLLAMA_PORT" "$model_tag"; then
                log "  [SMOKE FAIL] $MODEL_NAME — skipping"
                SKIPPED_MODELS+=("ollama:$MODEL_NAME (smoke failed)")
                ollama stop "$model_tag" 2>/dev/null || true
                sleep 5; continue
            fi
            log "  [SMOKE PASS]"
        fi

        BENCH_MODELS+=("ollama:$i")
        echo "MODEL_START:ollama:${MODEL_NAME}:${MODEL_QUANT}:$(date +%s)" >> "$TIMING_LOG"

        local RUN_NUM=0
        local TOTAL_RUNS=$(( ${#GEN_LENGTHS[@]} * ${#PROMPT_LENGTHS[@]} ))
        for GEN in "${GEN_LENGTHS[@]}"; do
            for CTX in "${PROMPT_LENGTHS[@]}"; do
                RUN_NUM=$((RUN_NUM + 1))
                local ARTIFACT_DIR="$BASE_ARTIFACT/ollama/$MODEL_NAME/gen${GEN}/ctx${CTX}"
                mkdir -p "$ARTIFACT_DIR"
                [ -f "$ARTIFACT_DIR/profile_export_aiperf.json" ] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [RESUME SKIP]"; continue; }
                [[ -n "$ONLY_COMBO" && ",$ONLY_COMBO," != *",${GEN}:${CTX},"* ]] && \
                    { log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX gen=$GEN  [COMBO SKIP]"; continue; }
                log "  [$RUN_NUM/$TOTAL_RUNS] prompt=$CTX  gen=$GEN  reqs=$REQS"

                # Verify Ollama is still alive
                local code
                code=$(curl -s "http://localhost:$OLLAMA_PORT/api/tags" --max-time 5 \
                       -o /dev/null -w "%{http_code}" 2>/dev/null || true)
                if [ "$code" != "200" ]; then
                    log "  [!] Ollama not responding — aborting"
                    SKIPPED_MODELS+=("ollama:$MODEL_NAME (ollama died at gen=$GEN ctx=$CTX)")
                    break 2
                fi

                [ "$DRY_RUN" = 0 ] && start_stats_logger "$ARTIFACT_DIR/power.log"
                aiperf profile \
                    --model                         "$model_tag" \
                    --streaming \
                    --endpoint-type                 'chat' \
                    --url                           "http://localhost:$OLLAMA_PORT" \
                    --tokenizer                     "$MODEL_TOKENIZER" \
                    --synthetic-input-tokens-mean   "$CTX" \
                    --synthetic-input-tokens-stddev 0 \
                    --output-tokens-mean            "$GEN" \
                    --use-legacy-max-tokens \
                    --request-count                 "$REQS" \
                    --concurrency                   "$CONCURRENCY" \
                    --slice-duration                "$SLICE_DURATION" \
                    --random-seed                   "$RANDOM_SEED" \
                    --request-timeout-seconds       "$REQUEST_TIMEOUT" \
                    --artifact-dir                  "$ARTIFACT_DIR" \
                    || log "  aiperf failed (ctx=$CTX gen=$GEN)"
                [ "$DRY_RUN" = 0 ] && stop_stats_logger
                [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && { log "  Cooldown ${COOLDOWN_COMBO}s..."; sleep "$COOLDOWN_COMBO"; }
            done
        done

        echo "MODEL_END:ollama:${MODEL_NAME}:${MODEL_QUANT}:$(date +%s)" >> "$TIMING_LOG"
        log "Unloading $model_tag from CPU memory..."
        # OLLAMA_KEEP_ALIVE=0 ensures model is immediately evicted from CPU memory
        curl -s "http://localhost:$OLLAMA_PORT/api/generate" \
            -d "{\"model\":\"$model_tag\",\"keep_alive\":0,\"prompt\":\"\"}" \
            --max-time 10 -o /dev/null 2>/dev/null || true
        ollama stop "$model_tag" 2>/dev/null || true
        log "Cooling ${COOLDOWN_MODEL}s..."
        sleep "$COOLDOWN_MODEL"
    done

    # Restore Ollama to normal system operation
    log "Restoring Ollama daemon..."
    kill "$OLLAMA_BENCH_PID" 2>/dev/null || true
    sleep 2
    sudo systemctl start ollama 2>/dev/null || true
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

deactivate 2>/dev/null || true
[ "$DRY_RUN" = 0 ] && stop_stats_logger

[ "$DRY_RUN" = 1 ] && banner "Dry run complete" && exit 0
if [ "${#BENCH_MODELS[@]}" = 0 ]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════╗"
    echo "  ║  No models were benchmarked.                         ║"
    echo "  ╠══════════════════════════════════════════════════════╣"
    echo "  ║  Common causes:                                      ║"
    echo "  ║   • llama-server binary not found / not built        ║"
    echo "  ║   • All GGUF files missing from $GGUF_DIR"
    echo "  ║   • --only filter matched nothing                    ║"
    echo "  ║   • Server failed to start (OOM / port conflict)     ║"
    echo "  ╚══════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi

# ── Generate report.md ────────────────────────────────────────────────────────
banner "Generating report.md"

cat > "$REPORT_PY" << 'PYEOF'
import re, sys, json, os, glob
from datetime import datetime

base_dir   = sys.argv[1]
tegra_log  = sys.argv[2]
timing_log = sys.argv[3]
skipped_arg = sys.argv[4] if len(sys.argv) > 4 else ""
report_path = os.path.join(base_dir, "report.md")

skipped = [s for s in skipped_arg.split("||") if s] if skipped_arg else []

# ── PMIC power-log helpers ────────────────────────────────────────────────────
def _parse_tegrastats_lines(lines):
    pwr, cpu_t, gpu_t, tj_t = [], [], [], []
    for line in lines:
        pw_m  = re.search(r'VDD_CPU_GPU_CV (\d+)mW', line)
        cpu_m = re.search(r'cpu@([\d.]+)C', line)
        gpu_m = re.search(r'gpu@([\d.]+)C', line)
        tj_m  = re.search(r'tj@([\d.]+)C',  line)
        if pw_m:
            pwr.append(int(pw_m.group(1)) / 1000.0)
            if cpu_m: cpu_t.append(float(cpu_m.group(1)))
            if gpu_m: gpu_t.append(float(gpu_m.group(1)))
            if tj_m:  tj_t.append(float(tj_m.group(1)))
    if not pwr:
        return None, None, None, None
    return (
        sum(pwr) / len(pwr),
        sum(cpu_t) / len(cpu_t) if cpu_t else None,
        sum(gpu_t) / len(gpu_t) if gpu_t else None,
        max(tj_t) if tj_t else None,
    )

def read_tegrastats(path):
    try:
        return _parse_tegrastats_lines(open(path))
    except FileNotFoundError:
        return None, None, None, None

# ── Global power-log fallback (for old artifact dirs without per-combo files) ──
global_samples = []  # (timestamp, pw, cpu, gpu, tj)
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
            global_samples.append((ts, int(pw_m.group(1))/1000.0,
                float(cpu_m.group(1)) if cpu_m else None,
                float(gpu_m.group(1)) if gpu_m else None,
                float(tj_m.group(1))  if tj_m  else None))
except FileNotFoundError:
    pass

def power_in_window(t0, t1):
    w = [s for s in global_samples if t0 <= s[0] <= t1]
    if not w: return None, None, None, None
    cpu_t = [s[2] for s in w if s[2] is not None]
    gpu_t = [s[3] for s in w if s[3] is not None]
    tj_t  = [s[4] for s in w if s[4] is not None]
    return (sum(s[1] for s in w) / len(w),
            sum(cpu_t)/len(cpu_t) if cpu_t else None,
            sum(gpu_t)/len(gpu_t) if gpu_t else None,
            max(tj_t)             if tj_t  else None)

# ── Parse timing log — quant + fallback windows ────────────────────────────────
# FORMAT: MODEL_START:backend:name:quant:ts
model_quants  = {}
model_windows = {}
try:
    for line in open(timing_log):
        line = line.strip()
        if line.startswith("MODEL_START:"):
            parts = line.split(":", 4)
            if len(parts) == 5:
                _, backend, name, quant, ts = parts
                key = f"{backend}:{name}"
                model_quants[key] = quant
                model_windows.setdefault(key, {})["start"] = float(ts)
        elif line.startswith("MODEL_END:"):
            parts = line.split(":", 4)
            if len(parts) == 5:
                _, backend, name, _, ts = parts
                model_windows.setdefault(f"{backend}:{name}", {})["end"] = float(ts)
except FileNotFoundError:
    pass

# ── Discover result files ─────────────────────────────────────────────────────
# Artifact structure: <base>/<backend>/<model>/gen<G>/ctx<C>/profile_export_aiperf.json
# Power: per-combo power.log preferred; falls back to global power.log+window for old runs
# for old artifact dirs that predate per-combo logging.
model_thermal = {}  # key -> {"power": [], "cpu": [], "gpu": [], "tj": []}
results = []
for json_path in sorted(glob.glob(f"{base_dir}/**/profile_export_aiperf.json", recursive=True)):
    rel   = os.path.relpath(json_path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) < 5: continue
    backend    = parts[0]  # llamacpp or ollama
    model_name = parts[1]
    gen = int(re.sub(r'\D', '', parts[2]))
    ctx = int(re.sub(r'\D', '', parts[3]))
    try: d = json.load(open(json_path))
    except: continue
    def g(key, stat="avg"): return (d.get(key, {}) or {}).get(stat)
    combo_tegra = os.path.join(os.path.dirname(json_path), "power.log")
    avg_pw, avg_cpu, avg_gpu, peak_tj = read_tegrastats(combo_tegra)
    win_key = f"{backend}:{model_name}"
    if avg_pw is None and global_samples:
        win = model_windows.get(win_key, {})
        avg_pw, avg_cpu, avg_gpu, peak_tj = power_in_window(win.get("start", 0), win.get("end", 9e18))
    mt = model_thermal.setdefault(win_key, {"power": [], "cpu": [], "gpu": [], "tj": []})
    if avg_pw  is not None: mt["power"].append(avg_pw)
    if avg_cpu is not None: mt["cpu"].append(avg_cpu)
    if avg_gpu is not None: mt["gpu"].append(avg_gpu)
    if peak_tj is not None: mt["tj"].append(peak_tj)
    tps = g("output_token_throughput_per_user")
    tok_j = (tps / avg_pw) if (tps and avg_pw) else None
    results.append({
        "backend": backend, "model": model_name,
        "quant": model_quants.get(win_key, "?"),
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

thermal = {}
all_model_keys = set(model_thermal.keys()) | set(model_windows.keys())
for key in all_model_keys:
    mt = model_thermal.get(key, {"power": [], "cpu": [], "gpu": [], "tj": []})
    avg_pw  = sum(mt["power"]) / len(mt["power"]) if mt["power"] else None
    avg_cpu = sum(mt["cpu"])   / len(mt["cpu"])   if mt["cpu"]   else None
    avg_gpu = sum(mt["gpu"])   / len(mt["gpu"])   if mt["gpu"]   else None
    peak_tj = max(mt["tj"])                        if mt["tj"]    else None
    if avg_pw is None and global_samples:
        win = model_windows.get(key, {})
        avg_pw, avg_cpu, avg_gpu, peak_tj = power_in_window(win.get("start", 0), win.get("end", 9e18))
    thermal[key] = {"avg_pw": avg_pw, "avg_cpu": avg_cpu, "avg_gpu": avg_gpu,
                    "peak_tj": peak_tj, "throttled": peak_tj is not None and peak_tj > 85}

best_tokj = {}
for r in results:
    k = f"{r['backend']}:{r['model']}"
    if r["tok_j"] is not None:
        if k not in best_tokj or r["tok_j"] > best_tokj[k]["tok_j"]: best_tokj[k] = r

lines = []
def L(s=""): lines.append(s)
def fmt(v, fs, fb="—"): return format(v, fs) if v is not None else fb

L("# Tiny LLM Benchmark — Raspberry Pi 5")
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
L(f"*Generated by `bench-non-reasoning.sh` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
L(f"*{len(results)} rows  |  {len(set(r['backend'] for r in results))} backends  |  {len(set(r['model'] for r in results))} models*")

with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  Report → {report_path}")
print(f"  {len(results)} rows across {len(backends_present)} backend(s)")
PYEOF

SKIPPED_STR=""
for s in "${SKIPPED_MODELS[@]}"; do SKIPPED_STR="${SKIPPED_STR}${s}||"; done

python3 "$REPORT_PY" "$BASE_ARTIFACT" "$TEGRA_LOG" "$TIMING_LOG" "$SKIPPED_STR"

banner "Done"
echo "  Artifacts : $BASE_ARTIFACT"
echo "  Report    : $BASE_ARTIFACT/report.md"
echo ""
echo "  Run again with Ollama:"
echo "    bash bench-non-reasoning.sh --backend ollama --resume $BASE_ARTIFACT"
echo ""
echo "  Dashboard (optional):"
echo "    source ~/venv/bin/activate"
echo "    AIPERF_DASHBOARD_HOST=0.0.0.0 aiperf plot $BASE_ARTIFACT --dashboard --port 8050"
echo ""
