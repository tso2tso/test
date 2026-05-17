#!/bin/bash
# =============================================================================
# Step 4: Distillation Training
# Function: Online distillation using PPO algorithm to compress Teacher knowledge into Student
# Optimization: Uses JSON cache for GT retrieval, no Neo4j service required
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}/distillation"

echo "=========================================="
echo "Distillation Training"
echo "=========================================="

# Optional parameters
STUDENT_PATH="${STUDENT_PATH:-outputs/sft/merged_model}"
DATA_PATH="${DATA_PATH:-data/processed/val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/minillm}"
TOTAL_ITERS="${TOTAL_ITERS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-5}"

echo "Student Model: ${STUDENT_PATH}"
echo "Training Data: ${DATA_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Total Iterations: ${TOTAL_ITERS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo ""
echo "📌 GT Retrieval: JSON cache (optimized, 1000+ times faster)"
echo ""

# Check if Teacher service is running
echo "Checking Teacher vLLM service..."
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ Teacher service is running"
else
    echo "❌ Warning: Teacher service is not running, please run 3_start_teacher.sh first"
    echo "Run in another terminal: bash scripts/3_start_teacher.sh"
    exit 1
fi
echo ""

python distill/trainer.py \
    --student-path "${STUDENT_PATH}" \
    --data-path "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --total-iters "${TOTAL_ITERS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}"

echo ""
echo "Distillation training complete!"
echo "Final model saved at: ${OUTPUT_DIR}/final"
