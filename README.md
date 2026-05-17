# Automotive Fault Diagnosis via Knowledge-Distilled LLM

This repository contains the official implementation for knowledge distillation of large language models applied to automotive fault diagnosis.

## Overview

We distill a 32B parameter teacher model enhanced with domain knowledge into a compact 7B student model, enabling efficient offline deployment for vehicle diagnostic systems.

**Key Features:**
- MiniLLM-based knowledge distillation with PPO
- Knowledge graph validation reward
- Long-tail balanced training data
- Multi-stage evaluation pipeline

## Repository Structure

```
├── Integrated_Data.json      # Diagnostic dataset
├── distillation/             # Training pipeline
│   ├── data/                 # Data preprocessing
│   ├── teacher/              # Teacher model (vLLM + DK)
│   ├── distill/              # Distillation core
│   └── train_sft.py          # SFT training
└── eval/                     # Evaluation module
    ├── evaluators/           # Threshold calibration evaluator
    └── benchmark_test/       # Benchmark tests
```

## Quick Start

### 1. Installation

```bash
pip install -r distillation/requirements.txt
```

### 2. Training

```bash
cd distillation
bash scripts/run_all.sh
```

### 3. Evaluation

```bash
cd eval
python run_eval.py
```

## Data Format

Input format:
```
ECUID: <ecu_id>, DTC: <dtc_code>, Trigger: <condition>, TimeCondition: <time>
```

Output format:
```json
{"FaultDescription": "...", "ServiceMeasures": "..."}
```
