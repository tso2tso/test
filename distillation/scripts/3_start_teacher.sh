#!/bin/bash
# =============================================================================
# Step 3: Start vLLM Teacher Service
# Function: Deploy Qwen3-32B as Teacher model using vLLM
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}/distillation"

echo "=========================================="
echo "Starting vLLM Teacher Service"
echo "=========================================="

# Optional parameters
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.6}"
MAX_LEN="${MAX_LEN:-2048}"

echo "Model: ${MODEL}"
echo "Address: http://${HOST}:${PORT}"
echo "Tensor Parallel: ${TP_SIZE}"
echo "GPU Utilization: ${GPU_UTIL}"
echo "Max Sequence Length: ${MAX_LEN}"
echo ""

python teacher/vllm_server.py \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${TP_SIZE}" \
    --gpu-util "${GPU_UTIL}" \
    --max-len "${MAX_LEN}"
