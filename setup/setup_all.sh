#!/usr/bin/env bash
#
# Convenience driver: build all three isolated environments with uv.
#
# The three environments are intentionally SEPARATE — the speculators training
# stack and the vllm serving stack have conflicting dependencies and must not
# share a venv (per the homework instructions).
#
# Run this ON THE GPU BOX (Linux + NVIDIA H100). On macOS the CUDA-only
# packages (vllm, flash-attn, CUDA torch) have no wheels and will fail.
#
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "############################################################"
echo "# 1/3  speculators_venv  (EAGLE-3 training)"
echo "############################################################"
bash "${HERE}/setup_speculators_venv.sh"

echo "############################################################"
echo "# 2/3  vllm_venv  (serving + benchmark)"
echo "############################################################"
bash "${HERE}/setup_vllm_venv.sh"

echo "############################################################"
echo "# 3/3  comp_venv  (FP8 quantization)"
echo "############################################################"
bash "${HERE}/setup_comp_venv.sh"

echo
echo "All environments created. They live in the project root:"
echo "  speculators_venv/  vllm_venv/  comp_venv/"
echo "Do NOT submit the virtual environments."
