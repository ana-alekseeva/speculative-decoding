#!/bin/bash
# FP8 dynamic quantization of Qwen/Qwen3-8B with llmcompressor.
#
# Produces a separate compressed-tensors checkpoint (default
# ./Qwen3-8B-FP8-Dynamic) without touching the original BF16 model, so you can
# benchmark baseline vs. quantized serving side by side.
#
# Usage:
#   bash quantize_qwen3_8b_fp8.sh
#
# Reference:
# https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/README.md

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
SAVE_DIR="Qwen3-8B-FP8-Dynamic"
# =======================================

# Quantization runs in the dedicated compression venv (llmcompressor==0.12.0).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP_PY="$REPO_DIR/comp_venv/bin/python"

if [[ ! -x "$COMP_PY" ]]; then
    echo "ERROR: comp_venv not found at $COMP_PY"
    echo "Create it first:  bash setup/setup_comp_venv.sh"
    exit 1
fi

echo "=== FP8 dynamic quantization: $MODEL -> $SAVE_DIR ==="
"$COMP_PY" "$REPO_DIR/quantize_fp8.py" \
    --model "$MODEL" \
    --save-dir "$REPO_DIR/$SAVE_DIR" \
    --ignore lm_head

echo "Done. Quantized model saved to $REPO_DIR/$SAVE_DIR (original $MODEL untouched)."
