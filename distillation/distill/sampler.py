"""
Online Sampling Module
Responsible for sampling responses from Student model and obtaining Teacher Logits
"""

import os
import sys
import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DistillConfig


@dataclass
class RolloutElement:
    """Single sampling result"""
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    mask: torch.Tensor
    student_logprobs: torch.Tensor
    teacher_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    ecuid: str
    dtc: str
    response_text: str


@dataclass 
class RolloutBatch:
    """Batch sampling results"""
    query_tensors: torch.Tensor
    response_tensors: torch.Tensor
    mask: torch.Tensor
    student_logprobs: torch.Tensor
    teacher_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    ecuids: List[str]
    dtcs: List[str]
    response_texts: List[str]


class OnlineSampler:
    """
    Online Sampler
    1. Sample from prompt pool
    2. Generate responses using Student model
    3. Get Teacher Logits (with few-shot examples)
    4. Compute rewards
    """
    
    def __init__(
        self,
        student_model,
        student_tokenizer,
        teacher_client,
        teacher_tokenizer,
        reward_calculator,
        config: DistillConfig = None,
    ):
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.teacher_client = teacher_client
        
        self._few_shot_cache = None
        self.teacher_tokenizer = teacher_tokenizer
        self.reward_calculator = reward_calculator
        self.config = config or DistillConfig()
        
        self.device = next(student_model.parameters()).device
    
    @property
    def few_shot_examples(self) -> str:
        """Lazy-load few-shot examples"""
        if self._few_shot_cache is None:
            self._few_shot_cache = self._load_few_shot_examples()
            if self._few_shot_cache:
                print(f"✓ Loaded few-shot examples ({len(self._few_shot_cache)} chars)")
        return self._few_shot_cache
    
    def _load_few_shot_examples(self, num_examples: int = 3) -> str:
        """
        Load complete few-shot examples from Integrated_Data.json
        数据流: Integrated_Data.json -> few-shot examples
        """
        try:
            from config import TEACHER_GT_CONFIG
            json_path = TEACHER_GT_CONFIG.get("json_cache_path", "")
            
            if json_path and os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data and len(data) > 0:
                    import random
                    sample_pool = data[:min(200, len(data))]
                    if len(sample_pool) >= num_examples:
                        selected = random.sample(sample_pool, num_examples)
                    else:
                        selected = sample_pool
                    
                    examples = []
                    for idx, item in enumerate(selected, 1):
                        input_text = item.get("input", "")
                        output_obj = item.get("output", {})
                        output_text = json.dumps(output_obj, ensure_ascii=False)
                        
                        example = f"""### Example {idx}:
Input: {input_text}
Output: {output_text}
"""
                        examples.append(example)
                    
                    return "\n".join(examples) + "\n"
            
            return self._get_default_few_shot_examples()
            
        except Exception as e:
            print(f"Failed to load few-shot examples: {e}")
            return self._get_default_few_shot_examples()
    
    def _get_default_few_shot_examples(self) -> str:
        """Default few-shot examples"""
        return """### Example 1:
Input: ECUID: 0087, DTC: 5D0C, Trigger: A fault in the control unit was detected., TimeCondition: 5 s
Output: {"FaultDescription": "The fault is: LRR: Internal sensor fault.", "ServiceMeasures": "To resolve this issue: If the fault is currently present and/or has occurred more than twice, renew LRR control unit.."}

### Example 2:
Input: ECUID: 0087, DTC: 5D0E, Trigger: Voltage below 9 V and engine not in starting phase., TimeCondition: 500 ms
Output: {"FaultDescription": "The fault is: LRR: Operating voltage.", "ServiceMeasures": "To resolve this issue: If the supply voltage fault memory entry is saved for several control units, a system fault has occurred in the vehicle (e.g. battery condition not OK). Carry out following test module (ABL): Energy diagnosis."}

### Example 3:
Input: ECUID: 0087, DTC: 5D0F, Trigger: A fault of the lens heating was detected., TimeCondition: 10 s
Output: {"FaultDescription": "The fault is: LRR: Lens heating.", "ServiceMeasures": "To resolve this issue: If the fault is currently present and/or has occurred more than twice, renew LRR control unit.."}

"""
    
    def _get_gt_for_query(self, ecuid: str, dtc: str) -> Optional[str]:
        """Get GT answer for current query if available"""
        try:
            if hasattr(self, 'reward_calculator') and self.reward_calculator:
                teacher_agent = getattr(self.reward_calculator, 'teacher_agent', None)
                if teacher_agent and hasattr(teacher_agent, 'gt_cache'):
                    gt_data = teacher_agent.gt_cache.query_by_ecuid_dtc(ecuid, dtc)
                    if gt_data:
                        return json.dumps({
                            "FaultDescription": gt_data.get("fault_description", ""),
                            "ServiceMeasures": gt_data.get("service_measures", ""),
                        }, ensure_ascii=False)
        except Exception as e:
            pass
        return None
    
    def sample_batch(
        self,
        prompts: List[Dict],
    ) -> RolloutBatch:
        """Sample a batch of prompts"""
        batch_size = len(prompts)
        
        # 1. Build prompt tokens
        prompt_texts = []
        ecuids = []
        dtcs = []
        
        for p in prompts:
            messages = [
                {"role": "system", "content": self.config.system_prompt if hasattr(self.config, 'system_prompt') else "You are a professional automotive fault diagnosis expert."},
                {"role": "user", "content": p["input"]}
            ]
            text = self.student_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(text)
            ecuids.append(p.get("ecuid", ""))
            dtcs.append(p.get("dtc", ""))
        
        # Tokenize prompts
        prompt_encodings = self.student_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.device)
        
        query_ids = prompt_encodings["input_ids"]
        query_mask = prompt_encodings["attention_mask"]
        
        # 2. Student generation
        def check_model_health():
            for name, param in self.student_model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    return False, name
            return True, None
        
        is_healthy, bad_param = check_model_health()
        if not is_healthy:
            print(f"Warning: Model parameter {bad_param} contains NaN/Inf! Attempting fix...")
            with torch.no_grad():
                for name, param in self.student_model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param.copy_(torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0))
        
        with torch.no_grad():
            temperature = max(self.config.temperature, 0.1)
            
            try:
                repetition_penalty = getattr(self.config, 'repetition_penalty', 1.1)
                
                student_outputs = self.student_model.generate(
                    input_ids=query_ids,
                    attention_mask=query_mask,
                    max_new_tokens=self.config.max_length - self.config.max_prompt_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.student_tokenizer.pad_token_id,
                    eos_token_id=self.student_tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            except RuntimeError as e:
                print(f"Sampling generation failed: {e}")
                print("Trying greedy decoding as fallback...")
                student_outputs = self.student_model.generate(
                    input_ids=query_ids,
                    attention_mask=query_mask,
                    max_new_tokens=min(50, self.config.max_length - self.config.max_prompt_length),
                    do_sample=False,
                    pad_token_id=self.student_tokenizer.pad_token_id,
                    eos_token_id=self.student_tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        
        full_ids = student_outputs.sequences
        response_ids = full_ids[:, query_ids.size(1):]
        
        response_texts = self.student_tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        
        # 3. Compute Student Logprobs
        with torch.no_grad():
            student_forward = self.student_model(
                input_ids=full_ids,
                attention_mask=(full_ids != self.student_tokenizer.pad_token_id).long(),
            )
            student_logits = student_forward.logits[:, query_ids.size(1)-1:-1, :]
            
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                print("Warning: student_logits contains NaN/Inf, processing")
                student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            student_logits = torch.clamp(student_logits, min=-100.0, max=100.0)
        
        # 4. Create mask
        mask = (response_ids != self.student_tokenizer.pad_token_id).float()
        
        # 5. Compute Student log probs
        student_logprobs = self._get_token_logprobs(
            student_logits, response_ids, mask
        )
        
        # 6. Get Teacher logprobs for each generated token
        teacher_logprobs = self._get_teacher_token_logprobs(
            prompt_texts, response_texts, response_ids, mask,
            ecuids=ecuids, dtcs=dtcs
        )
        
        # 7. Compute rewards
        fd_weights = None
        sm_weights = None
        if prompts and "fd_weight" in prompts[0]:
            fd_weights = [p.get("fd_weight", 1.0) for p in prompts]
            sm_weights = [p.get("sm_weight", 1.0) for p in prompts]
        
        class_weights = None
        if prompts and "class_weight" in prompts[0]:
            class_weights = [p.get("class_weight", 1.0) for p in prompts]
        
        reward_output = self.reward_calculator.compute_total_reward_v2(
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            response_ids=response_ids,
            mask=mask,
            student_responses=response_texts,
            ecuids=ecuids,
            dtcs=dtcs,
            class_weights=class_weights,
            fd_weights=fd_weights,
            sm_weights=sm_weights,
        )
        
        # 8. Compute advantages
        from .reward import get_advantages
        advantages = get_advantages(
            rewards=reward_output.total_reward,
            mask=mask,
            gamma=self.config.gamma,
        )
        
        return RolloutBatch(
            query_tensors=query_ids,
            response_tensors=response_ids,
            mask=mask,
            student_logprobs=student_logprobs,
            teacher_logprobs=teacher_logprobs,
            rewards=reward_output.total_reward,
            advantages=advantages,
            ecuids=ecuids,
            dtcs=dtcs,
            response_texts=response_texts,
        )
    
    def _get_teacher_token_logprobs(
        self,
        prompt_texts: List[str],
        response_texts: List[str],
        response_ids: torch.Tensor,
        mask: torch.Tensor,
        ecuids: List[str] = None,
        dtcs: List[str] = None,
    ) -> torch.Tensor:
        """
        Get Teacher's actual log probability for each generated token.
        
        Key improvements:
        1. Uses few-shot examples for task understanding
        2. Optionally includes GT as reference
        3. Uses vLLM token_logprobs as reward signal
        
        Returns: [batch, seq] teacher logprobs
        """
        batch_size = len(prompt_texts)
        max_response_len = response_ids.size(1)
        
        default_logprob = -5.0
        teacher_logprobs = torch.full(
            (batch_size, max_response_len),
            default_logprob,
            device=self.device,
            dtype=torch.float32
        )
        
        few_shot_prefix = self.few_shot_examples if hasattr(self, 'few_shot_examples') else ""
        
        success_count = 0
        for i, (prompt, response) in enumerate(zip(prompt_texts, response_texts)):
            try:
                gt_hint = ""
                if ecuids and dtcs and i < len(ecuids) and i < len(dtcs):
                    gt_answer = self._get_gt_for_query(ecuids[i], dtcs[i])
                    if gt_answer:
                        gt_hint = f"\n[Reference Answer]: {gt_answer}\n"
                
                teacher_prompt = few_shot_prefix + gt_hint + "### Current Query:\n" + prompt + response
                
                completion = self.teacher_client.completions.create(
                    model=self.config.teacher_model_name,
                    prompt=teacher_prompt,
                    max_tokens=1,
                    logprobs=1,
                    echo=True,
                )
                
                if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
                    logprobs_data = completion.choices[0].logprobs
                    
                    if hasattr(logprobs_data, 'token_logprobs') and logprobs_data.token_logprobs:
                        token_logprobs = logprobs_data.token_logprobs
                        
                        full_prefix = few_shot_prefix + gt_hint + "### Current Query:\n" + prompt
                        prefix_tokens = self.teacher_tokenizer.encode(full_prefix, add_special_tokens=False)
                        prefix_len = len(prefix_tokens)
                        
                        response_logprobs = token_logprobs[prefix_len:]
                        
                        for t, lp in enumerate(response_logprobs):
                            if t >= max_response_len:
                                break
                            if lp is not None:
                                lp = max(min(lp, 0.0), -20.0)
                                teacher_logprobs[i, t] = lp
                        
                        success_count += 1
                    
            except Exception as e:
                if i == 0:
                    print(f"Failed to get Teacher logprobs: {e}")
        
        if success_count == 0 and batch_size > 0:
            print(f"Warning: Failed to get Teacher logprobs for all {batch_size} samples")
        
        teacher_logprobs = teacher_logprobs * mask
        
        return teacher_logprobs
    
    def _get_teacher_logits(
        self,
        prompt_texts: List[str],
        response_texts: List[str],
        max_response_len: int,
        student_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """[Deprecated] Kept for compatibility"""
        if student_logits is not None:
            return student_logits.clone().detach()
        else:
            vocab_size = len(self.teacher_tokenizer)
            batch_size = len(prompt_texts)
            return torch.zeros(
                batch_size, max_response_len, vocab_size,
                device=self.device, dtype=torch.float32
            )
    
    def _get_token_logprobs(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for specified tokens"""
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        vocab_size = logits.size(-1)
        token_ids = torch.clamp(token_ids, min=0, max=vocab_size - 1)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = torch.nan_to_num(log_probs, nan=-100.0, posinf=0.0, neginf=-100.0)
        
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        token_log_probs = torch.nan_to_num(token_log_probs, nan=0.0, posinf=0.0, neginf=-100.0)
        
        return token_log_probs * mask


class PromptDataLoader:
    """Prompt data loader for online sampling"""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.current_idx = 0
        
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(self.data)
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self) -> List[Dict]:
        if self.current_idx >= len(self.data):
            raise StopIteration
        
        batch = self.data[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        return self._convert_batch(batch)
    
    def _convert_batch(self, batch: List[Dict]) -> List[Dict]:
        """Convert batch format"""
        prompts = []
        for item in batch:
            messages = item.get("messages", [])
            user_msg = ""
            ecuid = ""
            dtc = ""
            output_key = ""
            
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    if "ECUID:" in user_msg:
                        parts = user_msg.split(",")
                        for part in parts:
                            if "ECUID:" in part:
                                ecuid = part.split(":")[-1].strip()
                            elif "DTC:" in part:
                                dtc = part.split(":")[-1].strip()
                elif msg["role"] == "assistant":
                    try:
                        output_data = json.loads(msg["content"])
                        output_key = output_data.get("ServiceMeasures", "")
                    except:
                        output_key = msg["content"][:100]
            
            prompts.append({
                "input": user_msg,
                "ecuid": ecuid,
                "dtc": dtc,
                "output_key": output_key,
            })
        
        return prompts
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class BalancedPromptDataLoader:
    """
    Stratified sampling data loader by output category.
    
    Core idea:
    - Each batch contains samples from different output categories
    - Avoids multiple samples pointing to same output in one batch
    - Ensures gradient direction not dominated by high-frequency categories
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        seed: int = 42,
        class_weights_path: str = None,
    ):
        import random
        self.random = random.Random(seed)
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.batch_size = batch_size
        self.seed = seed
        
        # Group by output category
        self.output_groups: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.data):
            output_key = self._get_output_key(item)
            if output_key not in self.output_groups:
                self.output_groups[output_key] = []
            self.output_groups[output_key].append(idx)
        
        self.output_keys = list(self.output_groups.keys())
        print(f"[BalancedPromptDataLoader] {len(self.data)} samples, {len(self.output_keys)} output categories")
        
        # Load FD and SM class weights
        self.sm_weights = {}
        self.fd_weights = {}
        
        if class_weights_path and os.path.exists(class_weights_path):
            with open(class_weights_path, "r", encoding="utf-8") as f:
                weights_data = json.load(f)
            
            if "sm_weights" in weights_data and "fd_weights" in weights_data:
                self.sm_weights = weights_data["sm_weights"]
                self.fd_weights = weights_data["fd_weights"]
                print(f"[BalancedPromptDataLoader] Loaded SM weights ({len(self.sm_weights)} classes) and FD weights ({len(self.fd_weights)} classes)")
            else:
                self.sm_weights = weights_data
                print(f"[BalancedPromptDataLoader] Loaded {len(self.sm_weights)} SM class weights (legacy format)")
        
        self.current_step = 0
        self.total_steps = (len(self.data) + batch_size - 1) // batch_size
    
    def _get_output_key(self, item: Dict) -> str:
        """Extract output category key from data item"""
        messages = item.get("messages", [])
        for msg in messages:
            if msg["role"] == "assistant":
                try:
                    output_data = json.loads(msg["content"])
                    return output_data.get("ServiceMeasures", "")
                except:
                    return msg["content"][:100]
        return ""
    
    def __iter__(self):
        self.current_step = 0
        for key in self.output_keys:
            self.random.shuffle(self.output_groups[key])
        return self
    
    def __next__(self) -> List[Dict]:
        if self.current_step >= self.total_steps:
            raise StopIteration
        
        self.current_step += 1
        
        batch_indices = []
        available_keys = [k for k in self.output_keys if len(self.output_groups[k]) > 0]
        
        if len(available_keys) >= self.batch_size:
            selected_keys = self.random.sample(available_keys, self.batch_size)
            for key in selected_keys:
                idx = self.random.choice(self.output_groups[key])
                batch_indices.append(idx)
        else:
            for key in available_keys:
                idx = self.random.choice(self.output_groups[key])
                batch_indices.append(idx)
            
            while len(batch_indices) < self.batch_size:
                key = self.random.choice(available_keys)
                idx = self.random.choice(self.output_groups[key])
                batch_indices.append(idx)
        
        batch = [self.data[i] for i in batch_indices]
        
        return self._convert_batch(batch)
    
    def _convert_batch(self, batch: List[Dict]) -> List[Dict]:
        """Convert batch format with FD and SM class weight info"""
        prompts = []
        for item in batch:
            messages = item.get("messages", [])
            user_msg = ""
            ecuid = ""
            dtc = ""
            sm_key = ""
            fd_key = ""
            
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    if "ECUID:" in user_msg:
                        parts = user_msg.split(",")
                        for part in parts:
                            if "ECUID:" in part:
                                ecuid = part.split(":")[-1].strip()
                            elif "DTC:" in part:
                                dtc = part.split(":")[-1].strip()
                elif msg["role"] == "assistant":
                    try:
                        output_data = json.loads(msg["content"])
                        sm_key = output_data.get("ServiceMeasures", "")
                        fd_key = output_data.get("FaultDescription", "")
                    except:
                        sm_key = msg["content"][:100]
                        fd_key = ""
            
            sm_weight = self.sm_weights.get(sm_key, 1.0)
            fd_weight = self.fd_weights.get(fd_key, 1.0)
            
            prompts.append({
                "input": user_msg,
                "ecuid": ecuid,
                "dtc": dtc,
                "sm_key": sm_key,
                "fd_key": fd_key,
                "sm_weight": sm_weight,
                "fd_weight": fd_weight,
                "class_weight": (sm_weight + fd_weight) / 2,
            })
        
        return prompts
    
    def __len__(self):
        return self.total_steps
