#!/bin/bash
# =============================================================================
# MiniLLM Distillation Training Complete Pipeline Script
# Teacher: Qwen3-32B | Student: Qwen3-8B
# =============================================================================

set -e

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISTILL_DIR="${PROJECT_ROOT}/distillation"

echo "=========================================="
echo "MiniLLM Distillation Training Pipeline"
echo "=========================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Distillation Dir: ${DISTILL_DIR}"
echo ""

# =============================================================================
# Phase 1: Data Preprocessing
# =============================================================================
echo "=========================================="
echo "Phase 1: Data Preprocessing"
echo "=========================================="

cd "${DISTILL_DIR}"
python data/prepare_data.py

echo "Data preprocessing complete!"
echo ""

# =============================================================================
# Phase 2: SFT Training
# =============================================================================
echo "=========================================="
echo "Phase 2: SFT Training"
echo "=========================================="

python train_sft.py

echo "SFT training complete!"
echo ""

# =============================================================================
# Phase 3: Start vLLM Teacher Service
# =============================================================================
echo "=========================================="
echo "Phase 3: Start vLLM Teacher Service"
echo "=========================================="

echo "Please run the following command in another terminal to start Teacher service:"
echo ""
echo "  python teacher/vllm_server.py --model Qwen/Qwen3-32B"
echo ""
echo "After Teacher service is started, press Enter to continue..."
read -p ""

# =============================================================================
# Phase 4: MiniLLM Distillation Training
# =============================================================================
echo "=========================================="
echo "Phase 4: MiniLLM Distillation Training"
echo "=========================================="

python distill/trainer.py \
    --student-path outputs/sft/merged_model \
    --data-path data/processed/train.json \
    --output-dir outputs/minillm \
    --total-iters 5000 \
    --batch-size 8 \
    --lr 1e-6

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Final model saved at: ${DISTILL_DIR}/outputs/minillm/final"
