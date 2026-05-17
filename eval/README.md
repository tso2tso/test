# Model Evaluation

Evaluation module for assessing SFT and distilled model accuracy on automotive fault diagnosis.

## Evaluation Pipeline

Multi-stage evaluation with threshold calibration:

1. **Gate 1**: High-confidence fast track (Score ≥ 90, Similarity ≥ 0.8)
2. **Gate 2**: Hard entity matching (DTC/ECU validation)
3. **Gate 3**: Logic contradiction detection
4. **Gate 4**: Safety net fallback
5. **Step 2**: Structured extraction + online arbitration

## Quick Start

```bash
cd eval

# Evaluate SFT model
python run_eval.py --model_path ../distillation/outputs/sft/merged_model --num_samples 100

# Evaluate distilled model
python run_eval.py --model_path ../distillation/outputs/minillm/final_model --num_samples 100

# Custom test data
python run_eval.py --model_path <MODEL_PATH> --test_data /path/to/test.json

# Retrieval-only baseline: returns the top retrieved KG/data record
python run_eval.py --baseline retrieval_copy --num_samples 100

# Retrieval-augmented baseline: 7B/teacher inference with retrieved KG/data facts
python run_eval.py --baseline retrieval_rag --model_path ../distillation/outputs/sft/merged_model --num_samples 100
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Model directory path, required for `model` and `retrieval_rag` |
| `--test_data` | Test data JSON file |
| `--num_samples` | Number of test samples |
| `--eval_field` | Field to evaluate: `FaultDescription` / `ServiceMeasures` / `both` |
| `--use_vllm` | Use vLLM API for inference |
| `--baseline` | `model` / `retrieval_copy` / `retrieval_rag` |
| `--retrieval_corpus` | Retrieval corpus path, defaults to root `Integrated_Data.json` |
| `--retrieval_top_k` | Number of retrieved KG/data facts for retrieval baselines |
| `--allow_self_retrieval` | Allow exact test-record retrieval from corpus; disabled by default for fair evaluation |

## Retrieval Baselines

`retrieval_copy` is a non-parametric KG/RAG baseline. It retrieves the most similar diagnostic case from the corpus and directly evaluates the retrieved `FaultDescription` and `ServiceMeasures`.

`retrieval_rag` keeps the same model inference path but injects top-k retrieved KG/data facts into the prompt. This can be used as a simplified "7B + retrieval" or "teacher + KG/RAG at inference" comparison by pointing `--model_path` / `--use_vllm` to the corresponding model service.

## Configuration

Edit `config.py`:

- `LLM_EVALUATOR_CONFIG`: Scoring LLM (DeepSeek-V3 / GPT-4o recommended)
- `SEMANTIC_MODEL_NAME`: Local embedding model path

## Output

Results saved to `results/eval_results_<baseline>_<timestamp>/`:

```json
{
  "summary": {
    "FaultDescription": {"total": 100, "pass": 87, "accuracy": 87.0},
    "ServiceMeasures": {"total": 100, "pass": 82, "accuracy": 82.0},
    "overall": {"accuracy": 84.5}
  }
}
```
