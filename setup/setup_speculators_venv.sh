#!/usr/bin/env bash
#
# speculators_venv  — Data prep, hidden-state generation, EAGLE-3 draft-head training.
#
#   speculators : git tag v0.5.0, EDITABLE install
#   Python      : 3.12
#   Target HW   : Linux + NVIDIA H100 80GB (CUDA)
#
# Run this ON THE GPU BOX (not on macOS). It will NOT work on Apple Silicon —
# speculators pulls in CUDA-only deps (e.g. flash-attn) that have no macOS wheels.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV="speculators_venv"
PYVER="3.12"
REPO_DIR="speculators"
TAG="v0.5.0"

# ---- preflight -------------------------------------------------------------
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found — this venv targets CUDA/H100 and will not be usable here."
fi

# ---- venv ------------------------------------------------------------------
echo "==> Creating ${VENV} (Python ${PYVER}) with uv"
uv venv --python "${PYVER}" "${VENV}"
PY="${PROJECT_ROOT}/${VENV}/bin/python"

# ---- clone speculators @ tag ----------------------------------------------
if [ ! -d "${REPO_DIR}/.git" ]; then
  echo "==> Cloning speculators @ ${TAG}"
  git clone --depth 1 --branch "${TAG}" https://github.com/vllm-project/speculators.git "${REPO_DIR}"
else
  echo "==> ${REPO_DIR} already present; fetching/checking out ${TAG}"
  git -C "${REPO_DIR}" fetch --tags origin
  git -C "${REPO_DIR}" checkout "${TAG}"
fi

# ---- editable install ------------------------------------------------------
echo "==> Editable install of speculators into ${VENV}"
uv pip install --python "${PY}" -e "./${REPO_DIR}"

# flash-attn note: if the editable install fails to build flash-attn, retry it
# without build isolation AFTER torch is installed, e.g.:
#   uv pip install --python "${PY}" flash-attn --no-build-isolation

echo
echo "==> Done. Activate with:"
echo "    source ${VENV}/bin/activate"
echo "Reference workflow: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_offline"
