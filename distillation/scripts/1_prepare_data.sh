#!/bin/bash
# =============================================================================
# Step 1: Data Preprocessing
# Function: Data balancing + format conversion + train/val split
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}/distillation"

echo "=========================================="
echo "Data Preprocessing"
echo "=========================================="

python data/prepare_data.py

echo ""
echo "Processing complete! Generated files:"
echo "  - data/processed/sft_balanced.json (balanced full data)"
echo "  - data/processed/train.json (training set)"
echo "  - data/processed/val.json (validation set)"
