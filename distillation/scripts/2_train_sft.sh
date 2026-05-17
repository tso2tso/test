#!/bin/bash
# =============================================================================
# Step 2: SFT Training
# Function: Supervised fine-tuning on Qwen3-8B to learn JSON output format
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}/distillation"

echo "=========================================="
echo "SFT Training"
echo "=========================================="

# Optional parameters
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"

echo "Model: ${MODEL_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation: ${GRAD_ACCUM}"
echo "Learning Rate: ${LR}"
echo ""

python train_sft.py

echo ""
echo "SFT training complete!"
echo "Model saved at: ${OUTPUT_DIR}"
