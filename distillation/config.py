"""
MiniLLM Distillation Training Configuration
Teacher: Qwen3-32B | Student: Qwen3-8B
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

# ==================== Path Configuration ====================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths - unified to root Integrated_Data.json
DATA_CONFIG = {
    # Raw data from root directory
    # 数据流: Integrated_Data.json -> prepare_data.py -> processed/*.json
    "raw_data_path": os.path.join(PROJECT_ROOT, "Integrated_Data.json"),
    "processed_data_dir": os.path.join(PROJECT_ROOT, "distillation", "data", "processed"),
    "sft_data_path": os.path.join(PROJECT_ROOT, "distillation", "data", "processed", "sft_balanced.json"),
    "train_data_path": os.path.join(PROJECT_ROOT, "distillation", "data", "processed", "train.json"),
    "val_data_path": os.path.join(PROJECT_ROOT, "distillation", "data", "processed", "val.json"),
}

# Model paths
MODEL_CONFIG = {
    # Teacher model (for vLLM deployment)
    "teacher_model_name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "teacher_model_path": None,
    
    # Student model (for SFT and distillation)
    "student_model_name": "Qwen/Qwen2.5-7B-Instruct",
    "student_model_path": None,
    
    # Output directories
    "output_dir": os.path.join(PROJECT_ROOT, "distillation", "outputs"),
    "sft_output_dir": os.path.join(PROJECT_ROOT, "distillation", "outputs", "sft"),
    "distill_output_dir": os.path.join(PROJECT_ROOT, "distillation", "outputs", "minillm"),
}

# Neo4j configuration (inherited from phase 1)
NEO4J_CONFIG = {
    "uri": "neo4j://127.0.0.1:7687",
    "username": "neo4j",
    "password": os.environ.get("NEO4J_PASSWORD", "XXX"),
    "database": "aitest",
}

# Teacher GT retrieval configuration
TEACHER_GT_CONFIG = {
    "use_json_cache": True,
    "use_neo4j_fallback": False,
    # 数据流: 使用根目录的 Integrated_Data.json 作为 GT 缓存
    "json_cache_path": os.path.join(PROJECT_ROOT, "Integrated_Data.json"),
}

# vLLM Teacher service configuration
VLLM_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    # Optional local model path; prefer env var to avoid leaking local directories
    # Examples: "Qwen/Qwen2.5-32B-Instruct-AWQ" or "/path/to/local/model"
    "model": os.environ.get("VLLM_MODEL", MODEL_CONFIG["teacher_model_name"]),
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 2048,
    "dtype": "auto",
}


# ==================== Data Processing Configuration ====================

@dataclass
class DataProcessingConfig:
    """Data preprocessing configuration"""
    # Long-tail balancing (square root sampling strategy)
    max_samples_per_response: int = 3000
    min_samples_per_response: int = 5
    sqrt_sampling_coefficient: int = 10
    
    # Data split
    val_ratio: float = 0.01
    seed: int = 42
    
    # Prompt template
    system_prompt: str = """You are a professional automotive fault diagnosis expert. Based on the given ECU ID, DTC (Diagnostic Trouble Code), and trigger conditions, provide accurate fault descriptions and service measures.
Please output strictly in JSON format with two fields: FaultDescription and ServiceMeasures."""
    
    user_template: str = "ECUID: {ecuid}, DTC: {dtc}, Trigger: {trigger}, TimeCondition: {time_condition}"
    
    # Output format
    output_format: str = "json"


# ==================== SFT Training Configuration ====================

@dataclass
class SFTConfig:
    """SFT training configuration"""
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Focal Loss configuration
    focal_gamma: float = 2.0
    
    # Sequence length
    max_seq_length: int = 1024
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Save and logging
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Precision
    bf16: bool = False
    fp16: bool = True
    
    # DeepSpeed
    deepspeed_config: Optional[str] = None
    
    # Output
    output_dir: str = MODEL_CONFIG["sft_output_dir"]


# ==================== MiniLLM Distillation Configuration ====================

@dataclass
class DistillConfig:
    """MiniLLM distillation training configuration"""
    # Model
    student_model_path: str = os.path.join(MODEL_CONFIG["sft_output_dir"], "lora_weights")
    teacher_vllm_url: str = f"http://{VLLM_CONFIG['host']}:{VLLM_CONFIG['port']}/v1"
    teacher_model_name: str = VLLM_CONFIG["model"]
    
    # PPO training parameters
    ppo_epochs: int = 1
    num_rollouts_per_device: int = 128
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    
    # Learning rate
    learning_rate: float = 5e-7
    warmup_iters: int = 200
    total_iters: int = 1500
    
    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.15
    max_length: int = 768
    max_prompt_length: int = 512
    
    # LoRA configuration (distillation phase)
    use_lora_for_distill: bool = True
    distill_lora_r: int = 128
    distill_lora_alpha: int = 256
    
    # Reference model configuration
    use_reference_model: bool = True
    ref_model_update_freq: int = 0
    
    # Reward configuration
    kl_coef: float = 1
    reward_scaling: float = 10.0
    cliprange: float = 0.2
    cliprange_reward: float = 10.0
    gamma: float = 1.0
    
    # rev_kl weight
    rev_kl_weight: float = 0.3
    
    # Regularization
    single_step_reg: bool = True
    length_norm: bool = True
    
    # KG hard constraint reward
    use_kg_reward: bool = False
    kg_reward_weight: float = 5.0
    kg_penalty_weight: float = -2.0
    
    # ECU verification reward
    use_ecu_reward: bool = True
    ecu_reward_weight: float = 5.0
    ecu_penalty_weight: float = -10.0
    
    # Long-tail balancing strategy
    use_balanced_sampling: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 3.0
    
    # Save and logging
    save_interval: int = 100
    eval_interval: int = 100
    log_interval: int = 20
    
    # Precision
    bf16: bool = True
    fp16: bool = False
    
    # Output
    output_dir: str = MODEL_CONFIG["distill_output_dir"]


# ==================== Evaluation Configuration ====================

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "exact_match",
        "entity_recall",
        "json_valid",
        "rouge_l",
    ])
    
    # KG validation
    use_kg_validation: bool = True
    
    # Sampling
    num_samples: int = 1000
    batch_size: int = 8
    
    # Output
    results_dir: str = os.path.join(MODEL_CONFIG["output_dir"], "eval_results")


def ensure_dirs():
    """Ensure all necessary directories exist"""
    dirs = [
        DATA_CONFIG["processed_data_dir"],
        MODEL_CONFIG["output_dir"],
        MODEL_CONFIG["sft_output_dir"],
        MODEL_CONFIG["distill_output_dir"],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Configuration loaded successfully!")
    print(f"Raw data path: {DATA_CONFIG['raw_data_path']}")
    print(f"Teacher model: {MODEL_CONFIG['teacher_model_name']}")
    print(f"Student model: {MODEL_CONFIG['student_model_name']}")
