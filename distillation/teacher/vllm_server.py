"""
vLLM Teacher Model Service
Deploy Qwen3-32B as Teacher model using vLLM
"""

import os
import sys
import subprocess
import argparse
from modelscope import snapshot_download

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VLLM_CONFIG, MODEL_CONFIG


def start_vllm_server(
    model: str = None,
    host: str = None,
    port: int = None,
    tensor_parallel_size: int = None,
    gpu_memory_utilization: float = None,
    max_model_len: int = None,
    quantization: str = None,
):
    """Start vLLM OpenAI-compatible service"""
    
    # Use config or parameters
    model = model or VLLM_CONFIG["model"]
    host = host or VLLM_CONFIG["host"]
    port = port or VLLM_CONFIG["port"]
    tensor_parallel_size = tensor_parallel_size or VLLM_CONFIG["tensor_parallel_size"]
    gpu_memory_utilization = gpu_memory_utilization or VLLM_CONFIG["gpu_memory_utilization"]
    max_model_len = max_model_len or VLLM_CONFIG["max_model_len"]
    quantization = quantization or VLLM_CONFIG.get("quantization")
    
    # Use local model path if available
    if MODEL_CONFIG.get("teacher_model_path"):
        model = MODEL_CONFIG["teacher_model_path"]
    
    print("=" * 60)
    print("Starting vLLM Teacher Service")
    print("=" * 60)
    print(f"Model: {model}")
    
    # Download model using ModelScope (consistent with train_sft_ddp.py)
    model_id = model.replace("Qwen/", "qwen/")
    print(f"Downloading model from ModelScope: {model_id}")
    local_model_path = snapshot_download(model_id)
    print(f"Model download complete, local path: {local_model_path}")
    
    print(f"Address: http://{host}:{port}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Model Length: {max_model_len}")
    if quantization:
        print(f"Quantization Method: {quantization}")
    print("=" * 60)
    
    # Build command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", local_model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--dtype", "auto",
    ]
    
    # Add quantization parameter if specified
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    print(f"\nExecuting command: {' '.join(cmd)}\n")
    
    # Start service
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nService stopped")
    except Exception as e:
        print(f"Failed to start: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Start vLLM Teacher Service")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Service address")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel Size")
    parser.add_argument("--gpu-util", type=float, default=0.35, help="GPU memory utilization")
    parser.add_argument("--max-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (awq/gptq)")
    
    args = parser.parse_args()
    
    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_len,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
