#!/usr/bin/env bash
#
# comp_venv  — FP8 dynamic quantization of the verifier model.
#
#   llmcompressor : v0.12.0
#   Python        : 3.12
#   Target HW     : Linux + NVIDIA H100 80GB (CUDA)
#
# Run this ON THE GPU BOX (not on macOS). llmcompressor pulls in CUDA torch.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV="comp_venv"
PYVER="3.12"

# ---- preflight -------------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/"; exit 1; }
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found — llmcompressor FP8 quantization targets CUDA/H100."
fi

# ---- venv ------------------------------------------------------------------
echo "==> Creating ${VENV} (Python ${PYVER}) with uv"
uv venv --python "${PYVER}" "${VENV}"
PY="${PROJECT_ROOT}/${VENV}/bin/python"

# ---- install ---------------------------------------------------------------
echo "==> Installing llmcompressor==0.12.0 into ${VENV}"
uv pip install --python "${PY}" "llmcompressor==0.12.0"

echo
echo "==> Done. Activate with:"
echo "    source ${VENV}/bin/activate"
echo "FP8 reference: https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/README.md"
