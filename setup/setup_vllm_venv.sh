#!/usr/bin/env bash
#
# vllm_venv  — Serving + benchmark runtime (`vllm bench serve`).
#
#   vllm    : v0.20.0
#   fastapi : <0.137   (compatibility with the selected vLLM version)
#   Python  : 3.12
#   Target HW : Linux + NVIDIA H100 80GB (CUDA)
#
# Run this ON THE GPU BOX (not on macOS). vllm==0.20.0 ships Linux/CUDA wheels
# only — there is no macOS arm64 wheel, so `uv pip install` will fail here.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV="vllm_venv"
PYVER="3.12"

# ---- preflight -------------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/"; exit 1; }
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found — vllm targets CUDA/H100 and will not be usable here."
fi

# ---- venv ------------------------------------------------------------------
echo "==> Creating ${VENV} (Python ${PYVER}) with uv"
uv venv --python "${PYVER}" "${VENV}"
PY="${PROJECT_ROOT}/${VENV}/bin/python"

# ---- install ---------------------------------------------------------------
echo "==> Installing vllm==0.20.0 and fastapi<0.137 into ${VENV}"
uv pip install --python "${PY}" "vllm==0.20.0" "fastapi<0.137"

echo
echo "==> Done. Activate with:"
echo "    source ${VENV}/bin/activate"
echo
echo "Benchmark example (from the notebook):"
echo "    vllm bench serve \\"
echo "        --model Qwen/Qwen3-8B \\"
echo "        --dataset-name hf \\"
echo "        --max-concurrency 8 \\"
echo "        --dataset-path philschmid/mt-bench \\"
echo "        --num-prompts 80"
