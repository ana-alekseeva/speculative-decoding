#!/bin/bash
# Online Eagle3 Training Script
#
# Runs the full online training pipeline: data preparation, vLLM server launch,
# and training (with hidden states generated on-the-fly from the live server).
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/eagle3_qwen3_8b_sharegpt_online_5k.sh
#
# For a detailed walkthrough, see 
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train_eagle3_online/

### Example E2E run for Qwen3-8B on 5k samples from ShareGPT ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. This is a good sanity check when creating a drafter for a new
# target model.

# Timing (on 4x NVIDIA H100 80GB GPUs, DP=2)
# Data Preprocessing: 26 seconds
# vLLM Server Startup: 74 seconds (1 min 14 secs)
# Training (5 epochs): 942 seconds (15 mins 42 secs)
# Total: 1042 seconds (17 mins 22 secs)

# Results on SpecBench (80 prompts, 256 output tokens):
# acceptance rate: 14.88%
# acceptance length: 1.45
# per-position acceptance:
#   position 0: 34.36%
#   position 1: 9.00%
#   position 2: 1.27%
# output throughput: 143.37 tok/s

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output"
VLLM_PORT=8000
DRAFT_VOCAB_SIZE=32000
MAX_SAMPLES=3000
SEQ_LENGTH=2048
EPOCHS=5
LR=1e-4

# GPU assignments (online training needs separate GPUs for vLLM and training)
VLLM_GPUS="0"
TRAIN_GPUS="0"
NUM_TRAIN_GPUS=1

# Interpreters: this repo uses two separate virtualenvs.
#   speculators_venv -> data prep + training
#   vllm_venv        -> the patched vLLM server (extract_hidden_states)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC_PY="$REPO_DIR/speculators_venv/bin/python"
VLLM_PY="$REPO_DIR/vllm_venv/bin/python"
TORCHRUN="$REPO_DIR/speculators_venv/bin/torchrun"
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
"$SPEC_PY" speculators/scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" "$VLLM_PY" speculators/scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size 1 --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Generate hidden states offline (in speculators venv)
echo "=== Step 3: Generating hidden states (offline) ==="
"$SPEC_PY" speculators/scripts/data_generation_offline.py \
  --preprocessed-data ./output \
  --endpoint "http://localhost:${VLLM_PORT}/v1" \
  --output ./output/hidden_states \
  --max-samples 5000 \
  --concurrency 32 \
  --validate-outputs

echo "Done. Hidden states saved to ./output/hidden_states/"

# Offline training does not need the vLLM server, and it would otherwise hold
# ~15 GiB on GPU 0 (shared with training). Stop it now to free that memory.
echo "Stopping vLLM server before training (offline training reads cached states)..."
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
trap - EXIT

# Step 4: Train EAGLE-3 draft head OFFLINE on the precomputed hidden states
echo "=== Step 4: Training (offline) ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$TORCHRUN" \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    speculators/scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing skip \
    --save-best \
    --logger tensorboard

echo "Done. Best checkpoint: $OUTPUT_DIR/checkpoints/checkpoint_best (use this for serving/benchmarking)."