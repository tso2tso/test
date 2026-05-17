"""
Evaluation Configuration
Supports manual triggering for SFT or distilled model accuracy testing
"""
import os

# ==================== Path Configuration ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Default test data path - unified to distillation/data/processed
# 数据流: Integrated_Data.json -> prepare_data.py -> val.json
DEFAULT_TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "distillation", "data", "processed", "val.json")
DEFAULT_RETRIEVAL_CORPUS_PATH = os.path.join(PROJECT_ROOT, "Integrated_Data.json")

# ==================== LLM Evaluator Configuration ====================

# 将密钥填入下方字符串；若仓库会推送远端，勿提交真实密钥。
OPENROUTER_API_KEY = ""

LLM_EVALUATOR_CONFIG = {
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "api_key": OPENROUTER_API_KEY,
    "model_name": "deepseek/deepseek-v3.2",
    "temperature": 0.0,
    "timeout": 90,
}

# ==================== Search Arbitration LLM Configuration ====================
LLM_SAME_CONFIG = {
    "api_key": OPENROUTER_API_KEY,
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "model_name": "gpt-5-mini:online",
    "temperature": 0.16,
    "timeout": 90,
}

# ==================== Semantic Model Configuration ====================
# Optional local embedding model path (directory). Set `SEMANTIC_MODEL_PATH` if you want to use a local copy.
# If not set or not found, it will fall back to ModelScope / HuggingFace.
SEMANTIC_MODEL_NAME = os.environ.get("SEMANTIC_MODEL_PATH", "XXX")
SEMANTIC_MODEL_FALLBACK_MODELSCOPE = "intfloat/e5-mistral-7b-instruct"
SEMANTIC_MODEL_FALLBACK_HF = "intfloat/e5-mistral-7b-instruct"
USE_MODELSCOPE = True

# ==================== Inference Configuration ====================
INFERENCE_CONFIG = {
    "use_vllm": False,
    "vllm_url": "http://127.0.0.1:8000/v1/chat/completions",
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": False,
    "batch_size": 8,
}

RETRIEVAL_BASELINE_CONFIG = {
    "corpus_path": DEFAULT_RETRIEVAL_CORPUS_PATH,
    "top_k": 3,
    # Keep this true for fair evaluation when the test set is derived from the same corpus.
    "leave_one_out": True,
}

STUDENT_SYSTEM_PROMPT = """You are a professional automotive fault diagnosis expert. Based on the given ECU ID, DTC (Diagnostic Trouble Code), and trigger conditions, provide accurate fault descriptions and service measures.
Please output strictly in JSON format with two fields: FaultDescription and ServiceMeasures."""

# ==================== Test Configuration ====================
TEST_SAMPLE_SIZE = 1

# ==================== Output Configuration ====================
OUTPUT_FORMAT = "json"
SAVE_DETAILED_RESULTS = True


def get_semantic_model_path():
    """
    Get semantic model path.
    Priority: Local path > ModelScope > HuggingFace
    """
    if os.path.exists(SEMANTIC_MODEL_NAME):
        return SEMANTIC_MODEL_NAME
    
    print(f"[Config] Default semantic model not found: {SEMANTIC_MODEL_NAME}")
    
    if USE_MODELSCOPE:
        try:
            from modelscope import snapshot_download
            print(f"[Config] Downloading from ModelScope: {SEMANTIC_MODEL_FALLBACK_MODELSCOPE}")
            
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            model_path = snapshot_download(
                model_id=SEMANTIC_MODEL_FALLBACK_MODELSCOPE,
                cache_dir=cache_dir
            )
            print(f"[Config] ModelScope download complete: {model_path}")
            return model_path
            
        except ImportError:
            print("[Config] ModelScope SDK not installed, trying HuggingFace...")
        except Exception as e:
            print(f"[Config] ModelScope download failed: {e}, trying HuggingFace...")
    
    print(f"[Config] Using HuggingFace model: {SEMANTIC_MODEL_FALLBACK_HF}")
    return SEMANTIC_MODEL_FALLBACK_HF


def ensure_output_dir():
    """Ensure output directory exists."""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    return EVAL_OUTPUT_DIR
