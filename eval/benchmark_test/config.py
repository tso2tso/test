"""
Benchmark Test Configuration
Configuration for local fine-tuned model inference testing
"""
import os

# ==================== Path Configuration ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default: load base Qwen2.5-7B from Hugging Face Hub.
# For fine-tuned merged weights, set EVAL_LOCAL_MODEL_PATH or replace with a local dir.
LOCAL_MODEL_PATH = os.environ.get(
    "EVAL_LOCAL_MODEL_PATH",
    "Qwen/Qwen2.5-7B-Instruct",
)

# Base model path (required for LoRA adapter, set None to auto-detect from adapter_config.json)
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# Test data path - unified to distillation/data/processed
# 数据流: Integrated_Data.json -> prepare_data.py -> test_benchmark.json
DATA_PATH = os.path.join(PROJECT_ROOT, "distillation", "data", "processed", "test_benchmark.json")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ==================== Local Model Inference Configuration ====================
LOCAL_MODEL_CONFIG = {
    "model_path": LOCAL_MODEL_PATH,
    "base_model_path": BASE_MODEL_PATH,
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "device": "cuda",
    "torch_dtype": "float16",
    "trust_remote_code": True,
}

# ==================== LLM Evaluator Configuration ====================
LLM_EVALUATOR_CONFIG = {
    "api_url": "XXXXX",
    "api_key": "XXXXX",
    "model_name": "deepseek/deepseek-v3.2",
    "temperature": 0.0,
    "timeout": 90,
}

# ==================== Search Arbitration LLM Configuration ====================
LLM_SAME_CONFIG = {
    "api_key": "XXXX",
    "api_url": "XXXX",
    "model_name": "gpt-5-mini:online",
    "temperature": 0.16,
    "timeout": 90,
}

# ==================== Semantic Model Configuration ====================
SEMANTIC_MODEL_NAME = "XXX"
SEMANTIC_MODEL_FALLBACK = "intfloat/e5-mistral-7b-instruct"

# ==================== Test Configuration ====================
TEST_SAMPLE_SIZE = 10

# Multi-pair test configuration
MULTI_PAIR_CONFIG = {
    "num_groups": 5,
    "pairs_per_group": 3,
}

# Multi-turn dialogue test configuration
MULTI_TURN_CONFIG = {
    "num_sessions": 5,
    "turns_per_session": 3,
}

# Remote API configuration (reserved for comparison tests)
MODEL_CONFIG = {
    "api_url": "XXX",
    "api_key": "YOUR_API_KEY_HERE",
    "model_name": "qwen/qwen-2.5-7b-instruct",
    "timeout": 120,
}

# ==================== Utility Functions ====================
def get_semantic_model_path():
    """Get semantic model path with fallback."""
    if os.path.exists(SEMANTIC_MODEL_NAME):
        return SEMANTIC_MODEL_NAME
    print(f"[Config] Semantic model not found: {SEMANTIC_MODEL_NAME}")
    print(f"[Config] Using HuggingFace fallback: {SEMANTIC_MODEL_FALLBACK}")
    return SEMANTIC_MODEL_FALLBACK


def ensure_output_dir():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def check_local_model():
    """Check if model can be loaded: Hub id (org/name) or local checkpoint directory."""
    hub_style = (
        "/" in LOCAL_MODEL_PATH.replace("\\", "/")
        and not os.path.isabs(LOCAL_MODEL_PATH)
        and len([p for p in LOCAL_MODEL_PATH.replace("\\", "/").split("/") if p]) == 2
    )
    if hub_style and not os.path.isdir(LOCAL_MODEL_PATH):
        print(f"[Config] Using HuggingFace Hub model: {LOCAL_MODEL_PATH}")
        return True

    if not os.path.isdir(LOCAL_MODEL_PATH):
        print(f"[Warning] Local model path not found or not a directory: {LOCAL_MODEL_PATH}")
        return False

    adapter_config = os.path.join(LOCAL_MODEL_PATH, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"[Config] Detected LoRA adapter: {LOCAL_MODEL_PATH}")
        print(f"[Config] Base model: {BASE_MODEL_PATH}")
        return True

    config_file = os.path.join(LOCAL_MODEL_PATH, "config.json")
    if os.path.exists(config_file):
        print(f"[Config] Detected full model: {LOCAL_MODEL_PATH}")
        return True

    print("[Warning] No valid model files found (need config.json or adapter_config.json)")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("Benchmark Test Configuration")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Local model path: {LOCAL_MODEL_PATH}")
    print(f"Test data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Semantic model: {get_semantic_model_path()}")
    print(f"Local model check: {'Found' if check_local_model() else 'Not found'}")
    print("=" * 60)
