# Knowledge Distillation

Distilling knowledge from **Qwen-32B + Domain Knowledge** into lightweight **Qwen-7B** for offline automotive fault diagnosis.

## Project Structure

```
distillation/
├── config.py              # Configuration
├── data/
│   └── prepare_data.py    # Data preprocessing (balancing + formatting)
├── train_sft.py           # Supervised fine-tuning
├── teacher/
│   ├── vllm_server.py     # vLLM deployment
│   └── teacher_agent.py   # Teacher Agent (LLM + KG)
├── distill/
│   ├── reward.py          # Reward computation (Logits + KG)
│   ├── sampler.py         # Online sampling
│   └── trainer.py         # PPO-based distillation
└── scripts/               # Training scripts
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
bash scripts/run_all.sh

# Or run step by step:
bash scripts/1_prepare_data.sh   # Data preprocessing
bash scripts/2_train_sft.sh      # SFT training
bash scripts/3_start_teacher.sh  # Start Teacher (new terminal)
bash scripts/4_train_distill.sh  # Distillation
```

## Training Pipeline

| Stage | Description |
|-------|-------------|
| **Data Prep** | Long-tail balancing via sqrt sampling, format conversion |
| **SFT** | LoRA fine-tuning on Qwen-7B for JSON output format |
| **Teacher** | vLLM service with DK-enhanced context |
| **Distillation** | PPO with reverse KL + DK validation reward |

## Key Configuration

Edit `config.py`:

```python
MODEL_CONFIG = {
    "teacher_model_name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "student_model_name": "Qwen/Qwen2.5-7B-Instruct",
}
```

## Hardware Requirements

| Stage | GPU Memory |
|-------|------------|
| SFT (LoRA) | 24GB+ |
| Teacher (AWQ) | 20GB+ |
| Distillation | 80GB (A100) |

## References

- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [vLLM Documentation](https://docs.vllm.ai/)
