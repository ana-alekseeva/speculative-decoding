#!/bin/bash
# Task 4: Serve & benchmark four configurations on a single GPU.
#
#   1. Baseline            Qwen/Qwen3-8B (BF16)
#   2. Speculative         Qwen/Qwen3-8B + trained EAGLE-3 draft head
#   3. FP8                 Qwen3-8B-FP8-Dynamic
#   4. FP8 + speculative   Qwen3-8B-FP8-Dynamic + trained EAGLE-3 draft head
#
# For the two speculative configs the number of draft tokens is SWEPT
# (SPEC_SWEEP) so it can be tuned independently, per the task requirements.
#
# All runs share: same dataset (philschmid/mt-bench), fixed concurrency, fixed
# seed, prefix caching DISABLED. Each config is served, benchmarked, then torn
# down before the next one starts (sequential -> fits one GPU).
#
# Usage:
#   bash benchmark_task4.sh                 # full sweep, all four configs
#   SPEC_SWEEP="2" bash benchmark_task4.sh  # single draft-token value
#
# Results (bench stdout + /metrics) are written under ./results_task4/.

set -uo pipefail

# ============ Configuration ============
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_BIN="$REPO_DIR/vllm_venv/bin/vllm"

BASE_MODEL="Qwen/Qwen3-8B"
FP8_MODEL="$REPO_DIR/Qwen3-8B-FP8-Dynamic"
DRAFT_MODEL="$(readlink -f "$REPO_DIR/output/checkpoints/checkpoint_best")"

# Draft-token values to try for the speculative configs (tune from results).
SPEC_SWEEP="${SPEC_SWEEP:-1 2 3}"

# Benchmark settings (identical across all configs).
DATASET_NAME="hf"
DATASET_PATH="philschmid/mt-bench"
NUM_PROMPTS=80
CONCURRENCY=8
SEED=42

# Serving settings (identical across all configs).
PORT=8000
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.85
GPU="${GPU:-0}"

RESULTS_DIR="$REPO_DIR/results_task4"
HEALTH_TIMEOUT=600
# =======================================

mkdir -p "$RESULTS_DIR"
VLLM_PID=""

stop_server() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[INFO] Stopping vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
    # vLLM spawns EngineCore subprocesses that can outlive the parent and keep
    # holding the port/GPU. Reap any that are still bound to our port, then wait
    # for the port to actually free before the next config starts.
    pkill -9 -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    local waited=0
    while ss -tln 2>/dev/null | grep -q ":${PORT} "; do
        sleep 2; waited=$((waited + 2))
        (( waited >= 60 )) && { echo "[WARN] port ${PORT} still busy after 60s"; break; }
    done
}
trap stop_server EXIT

# start_server <model> <log_file> [spec_config_json]
start_server() {
    local model="$1" log="$2" spec="${3:-}"
    local args=(
        serve "$model"
        --seed "$SEED"
        --port "$PORT"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --no-enable-prefix-caching
    )
    [[ -n "$spec" ]] && args+=(--speculative-config "$spec")

    echo "[INFO] Launching: vllm ${args[*]}"
    CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" "${args[@]}" > "$log" 2>&1 &
    VLLM_PID=$!

    echo "[INFO] Waiting for server health (timeout ${HEALTH_TIMEOUT}s)..."
    local elapsed=0
    until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[ERROR] Server died during startup. Last 40 log lines:" >&2
            tail -n 40 "$log" >&2
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if (( elapsed >= HEALTH_TIMEOUT )); then
            echo "[ERROR] Server failed to become ready in ${HEALTH_TIMEOUT}s" >&2
            return 1
        fi
    done
    echo "[INFO] Server ready."
}

# run_bench <model> <label>
run_bench() {
    local model="$1" label="$2"
    local out="$RESULTS_DIR/${label}.txt"
    echo "[INFO] Benchmarking '$label'..."
    CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" bench serve \
        --model "$model" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --seed "$SEED" \
        --port "$PORT" 2>&1 | tee "$out"

    # Capture speculative-decoding acceptance counters from Prometheus metrics.
    curl -s "http://localhost:${PORT}/metrics" 2>/dev/null \
        | grep -iE "spec_decode" > "$RESULTS_DIR/${label}.metrics.txt" || true
    if [[ -s "$RESULTS_DIR/${label}.metrics.txt" ]]; then
        "$REPO_DIR/vllm_venv/bin/python" - "$RESULTS_DIR/${label}.metrics.txt" <<'PY' || true
import re, sys
text = open(sys.argv[1]).read()
def total(metric):
    m = re.findall(rf'^{metric}(?:_total)?\s+([0-9.eE+]+)$', text, re.M)
    return sum(float(x) for x in m) if m else None
draft = total("vllm:spec_decode_num_draft_tokens")
acc   = total("vllm:spec_decode_num_accepted_tokens")
if draft and acc is not None:
    rate = acc / draft
    print(f"  -> acceptance rate: {rate:.4f} "
          f"(accepted={acc:.0f}, draft={draft:.0f})")
PY
    fi
}

# config <label> <model> [spec_json]
config() {
    local label="$1" model="$2" spec="${3:-}"
    echo
    echo "==================== $label ===================="
    start_server "$model" "$RESULTS_DIR/${label}.server.log" "$spec" || { stop_server; return 1; }
    run_bench "$model" "$label"
    stop_server
}

spec_json() { # <draft_model> <num_tokens>
    printf '{"model": "%s", "num_speculative_tokens": %s, "method": "eagle3"}' "$1" "$2"
}

# ---- 1. Baseline ----------------------------------------------------------
config "1_baseline" "$BASE_MODEL"

# ---- 2. Speculative decoding (sweep draft tokens) -------------------------
for n in $SPEC_SWEEP; do
    config "2_spec_n${n}" "$BASE_MODEL" "$(spec_json "$DRAFT_MODEL" "$n")"
done

# ---- 3. FP8 quantization --------------------------------------------------
config "3_fp8" "$FP8_MODEL"

# ---- 4. FP8 + speculative decoding (sweep draft tokens) -------------------
for n in $SPEC_SWEEP; do
    config "4_fp8_spec_n${n}" "$FP8_MODEL" "$(spec_json "$DRAFT_MODEL" "$n")"
done

# ---- Summary --------------------------------------------------------------
echo
echo "==================== SUMMARY (Output token throughput) ===================="
for f in "$RESULTS_DIR"/*.txt; do
    [[ "$f" == *.metrics.txt ]] && continue
    label="$(basename "$f" .txt)"
    tput="$(grep -m1 'Output token throughput' "$f" | grep -oE '[0-9.]+' | tail -1)"
    acc="$(grep -m1 'acceptance rate' "${f%.txt}.metrics.txt" 2>/dev/null | grep -oE '[0-9.]+' | head -1)"
    printf "  %-22s %10s tok/s   %s\n" "$label" "${tput:-NA}" "${acc:+acc=$acc}"
done
echo "Full results in: $RESULTS_DIR/"
