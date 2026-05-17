"""
Usage:
    # Evaluate SFT model
    python run_eval.py --model_path ../outputs/sft/merged_model --num_samples 100
    
    # Evaluate distilled model
    python run_eval.py --model_path ../outputs/minillm/final_model --num_samples 100
    
    # Use custom test data
    python run_eval.py --model_path ../outputs/sft --test_data ./custom_test.json
    
    # Evaluate only ServiceMeasures field
    python run_eval.py --model_path ../outputs/sft --eval_field ServiceMeasures

    # Retrieval-only baseline (no generation model required)
    python run_eval.py --baseline retrieval_copy --num_samples 100

    # 7B/teacher + retrieval context at inference
    python run_eval.py --baseline retrieval_rag --model_path ../outputs/sft/merged_model --num_samples 100
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LLM_EVALUATOR_CONFIG, LLM_SAME_CONFIG,
    DEFAULT_TEST_DATA_PATH, INFERENCE_CONFIG, SAVE_DETAILED_RESULTS,
    get_semantic_model_path, ensure_output_dir,
    STUDENT_SYSTEM_PROMPT, DEFAULT_RETRIEVAL_CORPUS_PATH,
    RETRIEVAL_BASELINE_CONFIG
)
from evaluators import ThresholdCalibrationEvaluator


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after each write for real-time output"""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_generation_logger(log_dir: str) -> logging.Logger:
    """
    Setup dedicated logger for student model generation.
    
    Args:
        log_dir: Log save directory
        
    Returns:
        Configured Logger object
    """
    gen_logger = logging.getLogger("student_generation")
    gen_logger.setLevel(logging.INFO)
    gen_logger.handlers = []
    
    log_file = os.path.join(log_dir, "generation_log.txt")
    
    file_handler = FlushFileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s\n%(message)s\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    gen_logger.addHandler(file_handler)
    
    print(f"[Log] Student model generation log (real-time): {log_file}")
    print(f"[Hint] Use tail -f {log_file} for real-time monitoring")
    
    return gen_logger


def log_generation(
    logger: logging.Logger, 
    idx: int, 
    prompt: str, 
    model_output: str,
    parsed_output: Any,
    gt_data: Any
):
    """
    Log single generation details.
    
    Args:
        logger: Logger object
        idx: Sample index
        prompt: Input prompt
        model_output: Raw model output
        parsed_output: Parsed output
        gt_data: Ground Truth data
    """
    if not isinstance(parsed_output, dict):
        parsed_output = {"raw_output": str(parsed_output)}
    
    if not isinstance(gt_data, dict):
        gt_data = {"raw_output": str(gt_data)}
    
    pred_fd = parsed_output.get('FaultDescription', '(not extracted)')
    pred_sm = parsed_output.get('ServiceMeasures', '(not extracted)')
    gt_fd = gt_data.get('FaultDescription', '(none)')
    gt_sm = gt_data.get('ServiceMeasures', '(none)')
    
    log_msg = f"""
{'='*80}
 Sample #{idx}
{'='*80}

 [Input Prompt]
{prompt}

 [Model Raw Output]
{model_output}

 [Parsed Result]
  - FaultDescription: {pred_fd}
  - ServiceMeasures: {pred_sm}

 [Ground Truth]
  - FaultDescription: {gt_fd}
  - ServiceMeasures: {gt_sm}
{'='*80}
"""
    logger.info(log_msg)


class ModelInference:
    """Model inference wrapper class"""
    
    def __init__(self, model_path: str, use_vllm: bool = False, vllm_url: str = None):
        self.model_path = os.path.abspath(model_path)
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url
        self.model = None
        self.tokenizer = None
        
        if use_vllm:
            import requests
            self.requests = requests
            print(f"[Inference] Using vLLM API: {vllm_url}")
        else:
            self._load_model()
    
    def _load_model(self):
        """Load HuggingFace model"""
        print(f"[Inference] Loading model: {self.model_path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Try 4-bit quantization to save memory
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"[Inference] 4-bit quantization failed, trying normal load: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.model.eval()
            print(f"[Inference] Model loaded successfully")
            
        except Exception as e:
            print(f"[Inference] Model load failed: {e}")
            raise
    
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response"""
        if self.use_vllm:
            return self._generate_vllm(prompt, max_new_tokens)
        else:
            return self._generate_local(prompt, max_new_tokens)
    
    def _generate_vllm(self, prompt: str, max_new_tokens: int) -> str:
        """Generate via vLLM API"""
        try:
            payload = {
                "model": "student-model",
                "messages": [
                    {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": INFERENCE_CONFIG.get("temperature", 0.1),
            }
            resp = self.requests.post(self.vllm_url, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"[vLLM Error] {resp.status_code}: {resp.text}")
                return ""
        except Exception as e:
            print(f"[vLLM Exception] {e}")
            return ""
    
    def _generate_local(self, prompt: str, max_new_tokens: int) -> str:
        """Local model generation"""
        import torch
        
        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=INFERENCE_CONFIG.get("temperature", 0.1),
                do_sample=INFERENCE_CONFIG.get("do_sample", False),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Assistant:" in full_output:
            response = full_output.split("Assistant:")[-1].strip()
        elif "<|assistant|>" in full_output:
            response = full_output.split("<|assistant|>")[-1].strip()
        else:
            input_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response = full_output[input_len:].strip()
        
        return response


def normalize_output_fields(output: Any) -> Dict[str, str]:
    """Normalize raw KG fields to the evaluation schema."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return {"FaultDescription": output, "ServiceMeasures": ""}

    if not isinstance(output, dict):
        return {"FaultDescription": str(output), "ServiceMeasures": ""}

    normalized = dict(output)
    if "ServiceMeasures" not in normalized and "RepairMeasures" in normalized:
        normalized["ServiceMeasures"] = normalized.get("RepairMeasures", "")
    normalized.setdefault("FaultDescription", "")
    normalized.setdefault("ServiceMeasures", "")
    return normalized


def parse_diagnostic_input(input_str: str) -> Dict[str, str]:
    """Extract core diagnostic fields from a prompt-like input string."""
    fields = {"ECUID": "", "DTC": "", "Trigger": "", "TimeCondition": ""}
    for key in fields:
        pattern = rf"{key}\s*:\s*(.*?)(?=,\s*(?:ECU_Variant|ECUID|DTC|Trigger|TimeCondition)\s*:|$)"
        match = re.search(pattern, input_str, flags=re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()
    return fields


def build_prompt_from_input(input_data: Any) -> str:
    """Build the same user prompt format used by the model evaluator."""
    if isinstance(input_data, str):
        return input_data
    return (
        f"ECUID: {input_data.get('ECUID', '')}, "
        f"DTC: {input_data.get('DTC', '')}, "
        f"Trigger: {input_data.get('Trigger', '')}, "
        f"TimeCondition: {input_data.get('TimeCondition', '')}"
    )


def extract_prompt_and_output(sample: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Support both chat-format and input/output-format evaluation samples."""
    if "messages" in sample:
        messages = sample["messages"]
        prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "{}")
        return prompt, normalize_output_fields(assistant_msg)

    input_data = sample.get("input", sample)
    prompt = build_prompt_from_input(input_data)
    output_data = sample.get("output", sample)
    return prompt, normalize_output_fields(output_data)


class RetrievalIndex:
    """Lightweight lexical retriever for KG/RAG baselines without extra dependencies."""

    def __init__(
        self,
        corpus_path: str,
        top_k: int = 3,
        leave_one_out: bool = True,
    ):
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.leave_one_out = leave_one_out
        self.records = self._load_records(corpus_path)
        print(f"[Retrieval] Loaded {len(self.records)} records from: {corpus_path}")

    def _load_records(self, corpus_path: str) -> List[Dict[str, Any]]:
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for item in data:
            prompt, output = extract_prompt_and_output(item)
            parsed = parse_diagnostic_input(prompt)
            records.append({
                "input": prompt,
                "input_norm": self._normalize_text(prompt),
                "output": output,
                "fields": parsed,
                "tokens": self._tokenize(prompt),
            })
        return records

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_norm = self._normalize_text(query)
        query_fields = parse_diagnostic_input(query)
        query_tokens = self._tokenize(query)

        scored = []
        for record in self.records:
            if self.leave_one_out and record["input_norm"] == query_norm:
                continue

            score = self._score(query_fields, query_tokens, record)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "rank": rank,
                "score": round(score, 4),
                "input": record["input"],
                "output": record["output"],
                "fields": record["fields"],
            }
            for rank, (score, record) in enumerate(scored[:self.top_k], start=1)
        ]

    def _score(self, query_fields: Dict[str, str], query_tokens: set, record: Dict[str, Any]) -> float:
        record_fields = record["fields"]
        score = 0.0

        if query_fields.get("ECUID") and query_fields["ECUID"].lower() == record_fields.get("ECUID", "").lower():
            score += 8.0
        if query_fields.get("DTC") and query_fields["DTC"].lower() == record_fields.get("DTC", "").lower():
            score += 30.0
        if (
            query_fields.get("ECUID")
            and query_fields.get("DTC")
            and query_fields["ECUID"].lower() == record_fields.get("ECUID", "").lower()
            and query_fields["DTC"].lower() == record_fields.get("DTC", "").lower()
        ):
            score += 80.0

        overlap = query_tokens & record["tokens"]
        union = query_tokens | record["tokens"]
        if union:
            score += 10.0 * (len(overlap) / len(union))
        score += min(len(overlap), 10) * 0.2
        return score

    @staticmethod
    def format_context(retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "No relevant KG/RAG facts were retrieved."

        blocks = []
        for item in retrieved:
            output = item["output"]
            blocks.append(
                f"[Retrieved Case #{item['rank']} | score={item['score']}]\n"
                f"Input: {item['input']}\n"
                f"FaultDescription: {output.get('FaultDescription', '')}\n"
                f"ServiceMeasures: {output.get('ServiceMeasures', '')}"
            )
        return "\n\n".join(blocks)


class RetrievalBaselineInference:
    """Retrieval-only or retrieval-augmented inference wrapper."""

    def __init__(
        self,
        retriever: RetrievalIndex,
        mode: str,
        base_inference: Optional[ModelInference] = None,
    ):
        self.retriever = retriever
        self.mode = mode
        self.base_inference = base_inference
        self.last_retrieval: List[Dict[str, Any]] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.last_retrieval = self.retriever.retrieve(prompt)

        if self.mode == "retrieval_copy":
            if not self.last_retrieval:
                return json.dumps({"FaultDescription": "", "ServiceMeasures": ""}, ensure_ascii=False)
            return json.dumps(self.last_retrieval[0]["output"], ensure_ascii=False)

        retrieval_context = self.retriever.format_context(self.last_retrieval)
        augmented_prompt = f"""Original diagnostic request:
{prompt}

Retrieved KG/RAG facts:
{retrieval_context}

Use the retrieved facts as supporting evidence. Output strictly in JSON with two fields:
FaultDescription and ServiceMeasures."""
        return self.base_inference.generate(augmented_prompt, max_new_tokens=max_new_tokens)


def load_test_data(data_path: str, num_samples: int = None) -> List[Dict]:
    """Load test data"""
    print(f"[Data] Loading test data: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if num_samples and num_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, num_samples)
        print(f"[Data] Randomly sampled {num_samples} records")
    
    print(f"[Data] Loaded {len(data)} test records")
    return data


def parse_model_output(output: str) -> Dict[str, str]:
    """Parse model output JSON"""
    try:
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]
        
        result = json.loads(output.strip())
        if isinstance(result, dict):
            return normalize_output_fields(result)
        else:
            return {"raw_parsed": str(result)}
    except (json.JSONDecodeError, Exception):
        import re
        result = {}
        
        try:
            fd_match = re.search(r'"FaultDescription"\s*:\s*"([^"]*)"', output)
            if fd_match:
                result["FaultDescription"] = fd_match.group(1)
            
            sm_match = re.search(r'"ServiceMeasures"\s*:\s*"([^"]*)"', output)
            if not sm_match:
                sm_match = re.search(r'"RepairMeasures"\s*:\s*"([^"]*)"', output)
            if sm_match:
                result["ServiceMeasures"] = sm_match.group(1)
        except Exception:
            pass
        
        return result


def run_evaluation(
    model_path: str = None,
    test_data_path: str = None,
    num_samples: int = None,
    eval_field: str = "both",
    use_vllm: bool = False,
    vllm_url: str = None,
    baseline: str = "model",
    retrieval_corpus: str = None,
    retrieval_top_k: int = None,
    allow_self_retrieval: bool = False
):
    """
    Run evaluation pipeline.
    
    Args:
        model_path: Model path (SFT or distilled model)
        test_data_path: Test data path
        num_samples: Number of test samples
        eval_field: Evaluation field ("FaultDescription", "ServiceMeasures", "both")
        use_vllm: Whether to use vLLM API
        vllm_url: vLLM service URL
        baseline: "model", "retrieval_copy", or "retrieval_rag"
        retrieval_corpus: KG/RAG corpus path
        retrieval_top_k: Number of retrieved records
        allow_self_retrieval: Whether to allow exact query record retrieval
    """
    if baseline in {"model", "retrieval_rag"} and not model_path:
        raise ValueError("--model_path is required for model and retrieval_rag baselines")

    output_dir = ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"eval_results_{baseline}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    gen_logger = setup_generation_logger(result_dir)
    gen_logger.info(f"Evaluation started | Baseline: {baseline} | Model: {model_path} | Time: {timestamp}")
    
    if test_data_path is None:
        test_data_path = DEFAULT_TEST_DATA_PATH
    test_data = load_test_data(test_data_path, num_samples)
    
    print("\n" + "="*60)
    print("Initializing inference baseline...")
    print("="*60)
    if baseline == "model":
        inference = ModelInference(
            model_path=model_path,
            use_vllm=use_vllm,
            vllm_url=vllm_url or INFERENCE_CONFIG.get("vllm_url")
        )
    else:
        retrieval_corpus = retrieval_corpus or RETRIEVAL_BASELINE_CONFIG.get(
            "corpus_path", DEFAULT_RETRIEVAL_CORPUS_PATH
        )
        retrieval_top_k = retrieval_top_k or RETRIEVAL_BASELINE_CONFIG.get("top_k", 3)
        retriever = RetrievalIndex(
            corpus_path=retrieval_corpus,
            top_k=retrieval_top_k,
            leave_one_out=not allow_self_retrieval,
        )
        base_inference = None
        if baseline == "retrieval_rag":
            base_inference = ModelInference(
                model_path=model_path,
                use_vllm=use_vllm,
                vllm_url=vllm_url or INFERENCE_CONFIG.get("vllm_url")
            )
        inference = RetrievalBaselineInference(
            retriever=retriever,
            mode=baseline,
            base_inference=base_inference,
        )
    
    print("\n" + "="*60)
    print("Initializing BMW Master Evaluator...")
    print("="*60)
    semantic_model = get_semantic_model_path()
    evaluator = ThresholdCalibrationEvaluator(
        semantic_model_name=semantic_model,
        llm_api_url=LLM_EVALUATOR_CONFIG["api_url"],
        llm_api_key=LLM_EVALUATOR_CONFIG["api_key"],
        llm_model_name=LLM_EVALUATOR_CONFIG["model_name"],
        same_config=LLM_SAME_CONFIG,
        language="zh"
    )
    
    if eval_field == "both":
        fields_to_eval = ["FaultDescription", "ServiceMeasures"]
    else:
        fields_to_eval = [eval_field]
    
    print("\n" + "="*60)
    print(f"Starting evaluation | Baseline: {baseline} | Model: {model_path}")
    print(f"Test samples: {len(test_data)} | Eval fields: {fields_to_eval}")
    print("="*60 + "\n")
    
    results = {
        "metadata": {
            "baseline": baseline,
            "model_path": model_path,
            "test_data_path": test_data_path,
            "retrieval_corpus": retrieval_corpus if baseline != "model" else None,
            "retrieval_top_k": retrieval_top_k if baseline != "model" else None,
            "allow_self_retrieval": allow_self_retrieval if baseline != "model" else None,
            "num_samples": len(test_data),
            "eval_fields": fields_to_eval,
            "timestamp": timestamp,
        },
        "summary": {},
        "detailed_results": [],
        "failed_cases": []
    }
    
    stats = {field: {"total": 0, "pass": 0, "fail": 0} for field in fields_to_eval}
    
    for idx, sample in enumerate(tqdm(test_data, desc="Evaluation progress")):
        prompt, output_data = extract_prompt_and_output(sample)
        
        # Model inference
        try:
            model_output = inference.generate(prompt)
            pred_data = parse_model_output(model_output)
        except Exception as e:
            print(f"\n[Error] Sample {idx} inference failed: {e}")
            pred_data = {}
            model_output = f"[Inference failed] {str(e)}"
        
        log_generation(
            logger=gen_logger,
            idx=idx,
            prompt=prompt,
            model_output=model_output,
            parsed_output=pred_data,
            gt_data=output_data
        )
        
        pred_fd_short = (pred_data.get('FaultDescription', '') or '')[:50]
        pred_sm_short = (pred_data.get('ServiceMeasures', '') or '')[:50]
        tqdm.write(f"\n[#{idx}] FD: {pred_fd_short}...")
        tqdm.write(f"      SM: {pred_sm_short}...")
        
        sample_result = {
            "index": idx,
            "input": prompt,
            "model_output": model_output,
            "evaluations": {}
        }
        if hasattr(inference, "last_retrieval"):
            sample_result["retrieved_context"] = inference.last_retrieval
        
        for field in fields_to_eval:
            gt = output_data.get(field, "") or ""
            pred = pred_data.get(field, "") or ""
            
            if not gt:
                continue
            
            stats[field]["total"] += 1
            
            try:
                eval_result = evaluator.llm_binary_judgment(field, gt, pred)
                sample_result["evaluations"][field] = {
                    "gt": gt,
                    "pred": pred,
                    "judgment": eval_result["judgment"],
                    "score": eval_result["score"],
                    "sim": eval_result["sim"],
                    "logic": eval_result["logic"],
                    "reason": eval_result["reason"]
                }
                
                if eval_result["judgment"] == "符合":
                    stats[field]["pass"] += 1
                else:
                    stats[field]["fail"] += 1
                    results["failed_cases"].append({
                        "index": idx,
                        "field": field,
                        "input": prompt,
                        "gt": gt,
                        "pred": pred,
                        "reason": eval_result["reason"]
                    })
                    
            except Exception as e:
                print(f"\n[Error] Sample {idx} field {field} evaluation failed: {e}")
                stats[field]["fail"] += 1
        
        results["detailed_results"].append(sample_result)
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("Evaluation complete - Results Summary")
    print("="*60)
    
    for field in fields_to_eval:
        total = stats[field]["total"]
        passed = stats[field]["pass"]
        failed = stats[field]["fail"]
        accuracy = (passed / total * 100) if total > 0 else 0
        
        results["summary"][field] = {
            "total": total,
            "pass": passed,
            "fail": failed,
            "accuracy": round(accuracy, 2)
        }
        
        print(f"\n[{field}]")
        print(f"  Total: {total} | Pass: {passed} | Fail: {failed}")
        print(f"  Accuracy: {accuracy:.2f}%")
    
    # Compute overall accuracy
    total_all = sum(stats[f]["total"] for f in fields_to_eval)
    pass_all = sum(stats[f]["pass"] for f in fields_to_eval)
    overall_accuracy = (pass_all / total_all * 100) if total_all > 0 else 0
    
    results["summary"]["overall"] = {
        "total": total_all,
        "pass": pass_all,
        "accuracy": round(overall_accuracy, 2)
    }
    
    print(f"\n[Overall Accuracy]: {overall_accuracy:.2f}%")
    
    # Save results
    result_file = os.path.join(result_dir, "eval_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] Full results: {result_file}")
    
    if results["failed_cases"] and SAVE_DETAILED_RESULTS:
        failed_file = os.path.join(result_dir, "failed_cases.json")
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(results["failed_cases"], f, ensure_ascii=False, indent=2)
        print(f"[Saved] Failed cases: {failed_file}")
    
    # Log evaluation summary
    summary_msg = f"""
{'#'*80}
📊 Evaluation Complete Summary
{'#'*80}

Baseline: {baseline}
Model path: {model_path}
Test samples: {len(test_data)}
Eval fields: {fields_to_eval}

"""
    for field in fields_to_eval:
        summary_msg += f"""
[{field}]
  - Total: {stats[field]['total']}
  - Pass: {stats[field]['pass']}
  - Fail: {stats[field]['fail']}
  - Accuracy: {results['summary'][field]['accuracy']:.2f}%
"""
    
    summary_msg += f"""
[Overall Accuracy]: {overall_accuracy:.2f}%

Results saved to: {result_dir}
{'#'*80}
"""
    gen_logger.info(summary_msg)
    
    for handler in gen_logger.handlers:
        handler.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MiniLLM Distilled Model Evaluation Tool")
    parser.add_argument(
        "--baseline",
        type=str,
        default="model",
        choices=["model", "retrieval_copy", "retrieval_rag"],
        help=(
            "Evaluation baseline. 'model' is the original model-only inference; "
            "'retrieval_copy' returns the top retrieved KG/data record; "
            "'retrieval_rag' injects retrieved KG/data facts into model inference."
        )
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Model path (required for model and retrieval_rag baselines)"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        default=None,
        help="Test data path (JSON format, defaults to val.json)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=None,
        help="Number of test samples (default: all)"
    )
    parser.add_argument(
        "--eval_field", 
        type=str, 
        default="both",
        choices=["FaultDescription", "ServiceMeasures", "both"],
        help="Evaluation field"
    )
    parser.add_argument(
        "--use_vllm", 
        action="store_true",
        help="Use vLLM API for inference (requires vLLM service running)"
    )
    parser.add_argument(
        "--vllm_url", 
        type=str, 
        default=None,
        help="vLLM service URL"
    )
    parser.add_argument(
        "--retrieval_corpus",
        type=str,
        default=None,
        help="Retrieval corpus path (defaults to root Integrated_Data.json)"
    )
    parser.add_argument(
        "--retrieval_top_k",
        type=int,
        default=None,
        help="Number of retrieved KG/RAG records for retrieval baselines"
    )
    parser.add_argument(
        "--allow_self_retrieval",
        action="store_true",
        help="Allow exact test sample retrieval from the corpus (disabled by default for fair evaluation)"
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model_path,
        test_data_path=args.test_data,
        num_samples=args.num_samples,
        eval_field=args.eval_field,
        use_vllm=args.use_vllm,
        vllm_url=args.vllm_url,
        baseline=args.baseline,
        retrieval_corpus=args.retrieval_corpus,
        retrieval_top_k=args.retrieval_top_k,
        allow_self_retrieval=args.allow_self_retrieval
    )


if __name__ == "__main__":
    main()
