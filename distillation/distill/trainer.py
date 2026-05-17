"""
Distillation Trainer
PPO-based online distillation training
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from openai import OpenAI

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DistillConfig, MODEL_CONFIG, DATA_CONFIG, VLLM_CONFIG
from distill.reward import RewardCalculator, get_advantages
from distill.sampler import OnlineSampler, PromptDataLoader, BalancedPromptDataLoader, RolloutBatch
from teacher.teacher_agent import TeacherAgent


class PPOLoss:
    """
    PPO Loss with Focal Loss mechanism.
    
    Focal Loss reduces weight of "easy" samples (already learned well),
    focusing training on "hard" samples (usually low-frequency categories).
    
    KL Penalty Strategy:
    - If using Reference Model (external KL), no ratio-based KL here
    - If no Reference Model, use ratio-based approximate KL
    """
    
    def __init__(self, config: DistillConfig, use_external_kl: bool = False):
        self.config = config
        self.focal_gamma = getattr(config, 'focal_gamma', 2.0)
        self.use_focal = getattr(config, 'use_focal_loss', True)
        self.use_external_kl = use_external_kl
    
    def compute(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute PPO loss with optional Focal weights"""
        
        # Numerical stability
        logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=-100.0)
        old_logprobs = torch.nan_to_num(old_logprobs, nan=0.0, posinf=0.0, neginf=-100.0)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Compute ratio
        log_ratio = (logprobs - old_logprobs) * mask
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, min=1e-8, max=100.0)
        
        # PPO Clipped Loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.config.cliprange,
            1.0 + self.config.cliprange,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = torch.nan_to_num(pg_loss, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Conditional KL penalty
        if not self.use_external_kl:
            kl_penalty = (ratio - 1) - log_ratio
            kl_penalty = torch.nan_to_num(kl_penalty, nan=0.0, posinf=1.0, neginf=-1.0)
            kl_penalty = torch.clamp(kl_penalty, min=-5.0, max=5.0)
            pg_loss = pg_loss + self.config.kl_coef * kl_penalty
        
        # Focal Loss weights (optional)
        focal_weight = torch.ones_like(pg_loss)
        if self.use_focal:
            token_confidence = torch.exp(logprobs)
            token_confidence = torch.clamp(token_confidence, min=0.01, max=0.99)
            focal_weight = (1 - token_confidence) ** self.focal_gamma
            focal_weight = focal_weight / (focal_weight.mean() + 1e-8)
            focal_weight = torch.clamp(focal_weight, min=0.1, max=5.0)
        
        weighted_pg_loss = pg_loss * focal_weight
        
        n = mask.sum()
        loss = (weighted_pg_loss * mask).sum() / n if n > 0 else torch.tensor(0.0, device=pg_loss.device)
        
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.config.cliprange).float().mean().item()
            focal_weight_mean = focal_weight[mask.bool()].mean().item() if mask.sum() > 0 else 1.0
            
            if not torch.isfinite(torch.tensor(approx_kl)):
                approx_kl = 0.0
            if not torch.isfinite(torch.tensor(clip_frac)):
                clip_frac = 0.0
            if not torch.isfinite(torch.tensor(focal_weight_mean)):
                focal_weight_mean = 1.0
        
        stats = {
            "pg_loss": loss.item() if torch.isfinite(loss) else 0.0,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "ratio_mean": ratio.mean().item() if torch.isfinite(ratio.mean()) else 1.0,
            "focal_weight_mean": focal_weight_mean,
        }
        
        return loss, stats


class MiniLLMTrainer:
    """
    MiniLLM Distillation Trainer
    
    Key improvements (solving KL divergence explosion):
    1. Supports loading SFT stage LoRA adapter for continued training
    2. Creates frozen Reference Model for KL divergence computation
    3. Uses true KL penalty instead of ratio-based approximation
    """
    
    def __init__(
        self,
        config: DistillConfig = None,
        student_model_path: str = None,
    ):
        self.config = config or DistillConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Student model
        student_path = student_model_path or self.config.student_model_path
        print(f"Loading Student model: {student_path}")

        # Support two input types:
        # 1) Complete HF model directory
        # 2) PEFT LoRA adapter directory
        is_peft_adapter = os.path.isfile(os.path.join(student_path, "adapter_config.json"))
        if is_peft_adapter:
            from peft import PeftConfig, PeftModel
            peft_cfg = PeftConfig.from_pretrained(student_path)
            base_model_path = peft_cfg.base_model_name_or_path
            print(f"Detected LoRA adapter, base_model = {base_model_path}")
        else:
            base_model_path = student_path
        
        self.is_peft_adapter = is_peft_adapter
        self.base_model_path = base_model_path
        self.adapter_path = student_path if is_peft_adapter else None

        # Tokenizer from base model
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        # Use bfloat16 for better stability
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            print("Using bfloat16 precision (more stable)")
        else:
            model_dtype = torch.float32
            print("Using float32 precision (bfloat16 not supported)")
        
        self.model_dtype = model_dtype
        
        # Load Policy Model (trainable)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        if is_peft_adapter:
            from peft import PeftModel
            self.student_model = PeftModel.from_pretrained(
                self.student_model,
                student_path,
                is_trainable=True,
            )
            print("Loaded SFT LoRA adapter (will continue training on this adapter)")
            
            lora_trainable_count = 0
            for name, param in self.student_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    lora_trainable_count += 1
            print(f"Set {lora_trainable_count} LoRA parameters to trainable")
            
            if hasattr(self.student_model, "print_trainable_parameters"):
                self.student_model.print_trainable_parameters()
        
        use_lora = getattr(self.config, 'use_lora_for_distill', False)
        
        if use_lora and (not is_peft_adapter):
            from peft import LoraConfig, get_peft_model, TaskType
            
            distill_lora_r = getattr(self.config, 'distill_lora_r', 8)
            distill_lora_alpha = getattr(self.config, 'distill_lora_alpha', 16)
            
            lora_config = LoraConfig(
                r=distill_lora_r,
                lora_alpha=distill_lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.student_model = get_peft_model(self.student_model, lora_config)
            print(f"Distillation phase enabled new LoRA (r={distill_lora_r}, alpha={distill_lora_alpha})")
            self.student_model.print_trainable_parameters()
        elif not is_peft_adapter:
            print("Distillation phase using full fine-tuning (no LoRA)")
            for param in self.student_model.parameters():
                param.requires_grad = True
        
        # Create Reference Model (frozen, for KL divergence)
        self.ref_model = None
        use_ref_model = getattr(self.config, 'use_reference_model', True)
        
        if use_ref_model and is_peft_adapter:
            print("\nCreating Reference Model (frozen old policy)...")
            from peft import PeftModel
            self.ref_model = "shared"
            print("Reference Model: Using shared mode (disable adapter for base output)")
            
        elif use_ref_model and not is_peft_adapter:
            print("\nCreating Reference Model (full copy)...")
            import copy
            self.ref_model = copy.deepcopy(self.student_model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print("Reference Model: Created independent copy")
        else:
            print("Not using Reference Model (using sampling logprobs as baseline)")
        
        # Enable gradient checkpointing
        if is_peft_adapter or use_lora:
            base_for_ckpt = getattr(self.student_model, "base_model", self.student_model)
            if hasattr(base_for_ckpt, "model"):
                base_for_ckpt = base_for_ckpt.model
            if hasattr(base_for_ckpt, "enable_input_require_grads"):
                base_for_ckpt.enable_input_require_grads()
            if hasattr(base_for_ckpt, "gradient_checkpointing_enable"):
                base_for_ckpt.gradient_checkpointing_enable()
            print("Enabled gradient_checkpointing on base model")
        else:
            if hasattr(self.student_model, "enable_input_require_grads"):
                self.student_model.enable_input_require_grads()
            self.student_model.gradient_checkpointing_enable()
        
        self.student_model.train()
        
        # Check model health
        print("Checking model health...")
        for name, param in self.student_model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: Loaded model parameter {name} contains NaN/Inf!")
                raise ValueError(f"Model {student_path} is corrupted, please check SFT training results")
        print("Model health check passed ✓")
        
        # Create Teacher Client (vLLM)
        print(f"Connecting to Teacher vLLM service: {self.config.teacher_vllm_url}")
        self.teacher_client = OpenAI(
            base_url=self.config.teacher_vllm_url,
            api_key="EMPTY",
        )
        
        # Create Teacher Agent (with KG)
        self.teacher_agent = TeacherAgent(
            vllm_base_url=self.config.teacher_vllm_url,
        )
        
        # Create reward calculator
        self.reward_calculator = RewardCalculator(
            config=self.config,
            teacher_agent=self.teacher_agent,
        )
        
        # Create sampler
        self.sampler = OnlineSampler(
            student_model=self.student_model,
            student_tokenizer=self.student_tokenizer,
            teacher_client=self.teacher_client,
            teacher_tokenizer=self.student_tokenizer,
            reward_calculator=self.reward_calculator,
            config=self.config,
        )
        
        # PPO loss
        use_external_kl = (self.ref_model is not None)
        self.ppo_loss = PPOLoss(self.config, use_external_kl=use_external_kl)
        print(f"PPOLoss config: use_external_kl={use_external_kl}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=1e-6,
        )
        
        # Gradient Accumulation
        self.gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Learning rate scheduler
        effective_total_steps = self.config.total_iters // self.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_iters // self.gradient_accumulation_steps,
            num_training_steps=effective_total_steps,
        )
        
        # Training state
        self.global_step = 0
        self.accumulation_step = 0
        self.best_reward = float("-inf")
    
    def _get_ref_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        query_len: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get Reference Model log probabilities.
        
        Used for true KL divergence: KL = log π_policy - log π_ref
        """
        if self.ref_model is None:
            return None
        
        with torch.no_grad():
            if self.ref_model == "shared":
                if hasattr(self.student_model, "disable_adapter_layers"):
                    self.student_model.disable_adapter_layers()
                    outputs = self.student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    self.student_model.enable_adapter_layers()
                else:
                    return None
            else:
                outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            
            logits = outputs.logits[:, query_len-1:-1, :]
            logits = torch.clamp(logits, min=-100.0, max=100.0)
            
            log_probs = F.log_softmax(logits, dim=-1)
            ref_logprobs = torch.gather(
                log_probs, dim=-1,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            ref_logprobs = ref_logprobs * mask
            ref_logprobs = torch.nan_to_num(ref_logprobs, nan=0.0, posinf=0.0, neginf=-100.0)
        
        return ref_logprobs
    
    def train(self, data_path: str = None, class_weights_path: str = None):
        """Main training loop"""
        data_path = data_path or DATA_CONFIG["train_data_path"]
        
        if class_weights_path is None:
            class_weights_path = os.path.join(
                os.path.dirname(data_path), "class_weights.json"
            )
        
        use_balanced_sampling = getattr(self.config, 'use_balanced_sampling', True)
        
        if use_balanced_sampling and os.path.exists(class_weights_path):
            print(f"Using stratified sampling data loader (BalancedPromptDataLoader)")
            prompt_loader = BalancedPromptDataLoader(
                data_path=data_path,
                batch_size=self.config.batch_size,
                seed=42,
                class_weights_path=class_weights_path,
            )
        else:
            print(f"Using standard data loader (PromptDataLoader)")
            prompt_loader = PromptDataLoader(
                data_path=data_path,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Starting MiniLLM Distillation Training")
        print("=" * 60)
        print(f"Total iterations: {self.config.total_iters}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.config.batch_size * self.gradient_accumulation_steps}")
        print(f"PPO Epochs: {self.config.ppo_epochs}")
        print(f"Learning Rate: {self.config.learning_rate}")
        print("=" * 60)
        
        logging_stats = defaultdict(float)
        pbar = tqdm(total=self.config.total_iters, desc="Training")
        
        epoch = 0
        while self.global_step < self.config.total_iters:
            epoch += 1
            if use_balanced_sampling and os.path.exists(class_weights_path):
                prompt_loader = BalancedPromptDataLoader(
                    data_path=data_path,
                    batch_size=self.config.batch_size,
                    seed=42 + epoch,
                    class_weights_path=class_weights_path,
                )
            else:
                prompt_loader = PromptDataLoader(
                    data_path=data_path,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    seed=42 + epoch,
                )
            
            for prompts in prompt_loader:
                if self.global_step >= self.config.total_iters:
                    break
                
                # 1. Sampling
                t0 = time.time()
                with torch.no_grad():
                    rollout_batch = self.sampler.sample_batch(prompts)
                sample_time = time.time() - t0
                
                # 2. PPO Update (with gradient accumulation)
                t0 = time.time()
                self.accumulation_step += 1
                is_accumulation_complete = (self.accumulation_step % self.gradient_accumulation_steps == 0)
                
                ppo_stats = self._ppo_update(
                    rollout_batch, 
                    do_optimizer_step=is_accumulation_complete,
                    gradient_scale=1.0 / self.gradient_accumulation_steps,
                )
                update_time = time.time() - t0
                
                if is_accumulation_complete:
                    # Check model health
                    if self._check_model_health():
                        print("Error: Model weights corrupted (contains NaN/Inf)! Stopping training...")
                        print(f"Last normal step: {self.global_step}")
                        self._save_checkpoint(f"emergency_step_{self.global_step}")
                        pbar.close()
                        return
                    
                    self.global_step += 1
                    pbar.update(1)
                
                # 3. Record stats
                for k, v in ppo_stats.items():
                    logging_stats[k] += v
                logging_stats["sample_time"] += sample_time
                logging_stats["update_time"] += update_time
                
                # 4. Logging
                if is_accumulation_complete and self.global_step % self.config.log_interval == 0:
                    divisor = self.config.log_interval * self.gradient_accumulation_steps
                    avg_stats = {k: v / divisor for k, v in logging_stats.items()}
                    self._log_stats(avg_stats)
                    logging_stats = defaultdict(float)
                
                # 5. Evaluation
                if is_accumulation_complete and self.global_step % self.config.eval_interval == 0:
                    eval_reward = self._evaluate()
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        self._save_checkpoint("best")
                
                # 6. Save checkpoint
                if is_accumulation_complete and self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
        
        pbar.close()
        
        self._save_checkpoint("final")
        self.teacher_agent.close()
        
        print("\nTraining complete!")
        print(f"Best reward: {self.best_reward:.4f}")
        print(f"Model saved to: {self.config.output_dir}")
    
    def _ppo_update(
        self, 
        batch: RolloutBatch, 
        do_optimizer_step: bool = True,
        gradient_scale: float = 1.0,
    ) -> Dict:
        """
        PPO Update with gradient accumulation and Reference Model KL constraint.
        """
        all_stats = defaultdict(float)
        
        def check_tensor(t, name):
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"Warning: {name} contains NaN/Inf, processing")
                return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
            return t
        
        batch_advantages = check_tensor(batch.advantages, "advantages")
        batch_rewards = check_tensor(batch.rewards, "rewards")
        batch_student_logprobs = check_tensor(batch.student_logprobs, "student_logprobs")
        
        raw_model = self.student_model
        if hasattr(self.student_model, "base_model"):
            raw_model = self.student_model.base_model
        if hasattr(raw_model, "model"):
            raw_model = raw_model.model
        vocab_size = raw_model.config.vocab_size
        
        if (batch.response_tensors >= vocab_size).any():
            print(f"Warning: response_tensors contains tokens exceeding vocab size! Clipping to vocab_size-1")
            batch.response_tensors = torch.clamp(batch.response_tensors, max=vocab_size - 1)
        if (batch.response_tensors < 0).any():
            print(f"Warning: response_tensors contains negative tokens! Setting to 0")
            batch.response_tensors = torch.clamp(batch.response_tensors, min=0)
        
        if batch.mask.sum() == 0:
            print("Warning: batch.mask is all zeros, skipping this batch")
            return {"pg_loss": 0.0, "mean_reward": 0.0, "mean_advantage": 0.0, "kl_div": 0.0}
        
        full_input_ids = torch.cat([batch.query_tensors, batch.response_tensors], dim=1)
        full_attention_mask = torch.cat([
            (batch.query_tensors != self.student_tokenizer.pad_token_id).long(),
            batch.mask.long()
        ], dim=1)
        query_len = batch.query_tensors.size(1)
        
        ref_logprobs = self._get_ref_logprobs(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            response_ids=batch.response_tensors,
            query_len=query_len,
            mask=batch.mask,
        )
        
        valid_ppo_epochs = 0
        for ppo_epoch in range(self.config.ppo_epochs):
            outputs = self.student_model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
            )
            
            logits = outputs.logits[:, query_len-1:-1, :]
            logits = torch.clamp(logits, min=-100.0, max=100.0)
            
            log_probs = F.log_softmax(logits, dim=-1)
            current_logprobs = torch.gather(
                log_probs, dim=-1, 
                index=batch.response_tensors.unsqueeze(-1)
            ).squeeze(-1)
            current_logprobs = current_logprobs * batch.mask
            current_logprobs = check_tensor(current_logprobs, "current_logprobs")
            
            if ref_logprobs is not None:
                kl_div = (current_logprobs - ref_logprobs) * batch.mask
                kl_div = torch.clamp(kl_div, min=-10.0, max=10.0)
                mean_kl = kl_div.sum() / batch.mask.sum()
                kl_penalty = self.config.kl_coef * mean_kl
            else:
                kl_penalty = torch.tensor(0.0, device=self.device)
                mean_kl = torch.tensor(0.0, device=self.device)
            
            loss, stats = self.ppo_loss.compute(
                logprobs=current_logprobs,
                old_logprobs=batch_student_logprobs,
                advantages=batch_advantages,
                mask=batch.mask,
            )
            
            if ref_logprobs is not None:
                loss = loss + kl_penalty
                stats["kl_div"] = mean_kl.item()
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN/Inf, skipping this PPO epoch")
                self.optimizer.zero_grad()
                continue
            
            if loss.abs() > 100:
                print(f"Warning: Loss too large ({loss.item():.2f}), clipping to 100")
                loss = torch.clamp(loss, min=-100, max=100)
            
            epoch_gradient_scale = gradient_scale / self.config.ppo_epochs
            scaled_loss = loss * epoch_gradient_scale
            
            try:
                scaled_loss.backward()
            except RuntimeError as e:
                print(f"Warning: Backward failed: {e}")
                continue
            
            has_nan_grad = False
            for name, param in self.student_model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: PPO epoch {ppo_epoch} detected NaN gradient, skipping this epoch")
                self.optimizer.zero_grad()
                continue
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 0.5)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: Gradient norm is NaN/Inf, skipping this epoch")
                self.optimizer.zero_grad()
                continue
            
            if do_optimizer_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            valid_ppo_epochs += 1
            
            for k, v in stats.items():
                all_stats[k] += v
        
        if do_optimizer_step and valid_ppo_epochs > 0:
            self.scheduler.step()
        
        num_valid_epochs = max(1, valid_ppo_epochs)
        for k in all_stats:
            all_stats[k] /= num_valid_epochs
        
        mask_sum = batch.mask.sum()
        if mask_sum > 0:
            all_stats["mean_reward"] = (batch_rewards.sum() / mask_sum).item()
            all_stats["mean_advantage"] = (batch_advantages.sum() / mask_sum).item()
        else:
            all_stats["mean_reward"] = 0.0
            all_stats["mean_advantage"] = 0.0
        
        if "kl_div" not in all_stats:
            all_stats["kl_div"] = 0.0
        
        return dict(all_stats)
    
    def _evaluate(self) -> float:
        """Evaluate current model"""
        self.student_model.eval()
        
        val_loader = PromptDataLoader(
            data_path=DATA_CONFIG["val_data_path"],
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        total_reward = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, prompts in enumerate(val_loader):
                if i >= 10:
                    break
                
                batch = self.sampler.sample_batch(prompts)
                total_reward += batch.rewards.sum().item()
                total_samples += batch.mask.sum().item()
        
        self.student_model.train()
        
        avg_reward = total_reward / total_samples if total_samples > 0 else 0
        print(f"\n[Eval] Step {self.global_step}: avg_reward = {avg_reward:.4f}")
        
        return avg_reward
    
    def _check_model_health(self) -> bool:
        """Check if model weights are healthy (no NaN/Inf)"""
        for name, param in self.student_model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Detected corrupted parameter: {name}")
                return True
        return False
    
    def _log_stats(self, stats: Dict):
        """Print training statistics"""
        log_str = f"Step {self.global_step}"
        for k, v in stats.items():
            if isinstance(v, float):
                log_str += f" | {k}: {v:.4f}"
            else:
                log_str += f" | {k}: {v}"
        print(log_str)
        
        log_file = os.path.join(self.config.output_dir, "train_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint"""
        save_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        self.student_model.save_pretrained(save_dir)
        self.student_tokenizer.save_pretrained(save_dir)
        
        state = {
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(state, os.path.join(save_dir, "trainer_state.pt"))
        
        print(f"Checkpoint saved to: {save_dir}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MiniLLM Distillation Training")
    parser.add_argument("--student-path", type=str, help="SFT Student model path")
    parser.add_argument("--data-path", type=str, help="Training data path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--total-iters", type=int, default=None, help="Total iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    
    args = parser.parse_args()
    
    config = DistillConfig()
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.total_iters:
        config.total_iters = args.total_iters
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    trainer = MiniLLMTrainer(
        config=config,
        student_model_path=args.student_path,
    )
    
    trainer.train(data_path=args.data_path)


if __name__ == "__main__":
    main()
