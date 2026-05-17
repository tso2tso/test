"""
Reward Calculation Module
Combines Teacher Logits and KG validation for comprehensive reward
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DistillConfig


@dataclass
class RewardOutput:
    """Reward calculation result"""
    total_reward: torch.Tensor  # [batch_size, seq_len]
    logit_reward: torch.Tensor  # Reward from Teacher Logits
    kg_reward: torch.Tensor     # Reward from KG validation
    rev_kl: torch.Tensor        # Reverse KL divergence
    info: Dict                  # Additional info


class RewardCalculator:
    """
    Reward Calculator
    Combines Teacher Logits and KG validation
    """
    
    def __init__(
        self,
        config: DistillConfig = None,
        teacher_agent = None,
    ):
        self.config = config or DistillConfig()
        self.teacher_agent = teacher_agent
    
    def compute_logit_reward(
        self,
        student_logits: torch.Tensor,  # [batch, seq, vocab]
        teacher_logits: torch.Tensor,  # [batch, seq, vocab]
        response_ids: torch.Tensor,    # [batch, seq]
        mask: torch.Tensor,            # [batch, seq]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logit-based reward (reverse KL divergence).
        
        R_t = log P_teacher(y_t | y_{<t}, x) - log P_student(y_t | y_{<t}, x)
        
        Returns: (reward, rev_kl)
        """
        # Numerical stability: handle NaN/Inf
        if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
            print("Warning: student_logits contains NaN/Inf, clipping")
            student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
            print("Warning: teacher_logits contains NaN/Inf, clipping")
            teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Temperature scaling
        student_logits = student_logits / self.config.temperature
        teacher_logits = teacher_logits / self.config.temperature
        
        # Clip logits to prevent softmax overflow
        student_logits = torch.clamp(student_logits, min=-100.0, max=100.0)
        teacher_logits = torch.clamp(teacher_logits, min=-100.0, max=100.0)
        
        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        
        # Get log probabilities for generated tokens
        response_ids_expanded = response_ids.unsqueeze(-1)
        
        student_token_log_probs = torch.gather(
            student_log_probs, dim=-1, index=response_ids_expanded
        ).squeeze(-1)
        
        teacher_token_log_probs = torch.gather(
            teacher_log_probs, dim=-1, index=response_ids_expanded
        ).squeeze(-1)
        
        # Reverse KL: log P_teacher - log P_student
        rev_kl = teacher_token_log_probs - student_token_log_probs
        
        # Reward = Teacher log prob + entropy bonus
        probs = torch.softmax(student_logits, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        student_entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        
        reward = teacher_token_log_probs + student_entropy
        
        # Numerical stability: clip reward values
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        rev_kl = torch.nan_to_num(rev_kl, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply mask
        reward = reward * mask
        rev_kl = rev_kl * mask
        
        return reward, rev_kl
    
    def compute_kg_reward(
        self,
        student_responses: List[str],
        ecuids: List[str],
        dtcs: List[str],
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Compute KG validation reward.
        
        Returns: (kg_reward, info_list)
        """
        if not self.config.use_kg_reward or self.teacher_agent is None:
            batch_size = len(student_responses)
            return torch.zeros(batch_size), [{}] * batch_size
        
        kg_rewards = []
        info_list = []
        
        for response, ecuid, dtc in zip(student_responses, ecuids, dtcs):
            reward, info = self.teacher_agent.compute_reward(
                student_response=response,
                ecuid=ecuid,
                dtc=dtc,
                config=self.config,
            )
            kg_rewards.append(reward)
            info_list.append(info)
        
        return torch.tensor(kg_rewards), info_list
    
    def compute_ecu_reward(
        self,
        student_responses: List[str],
        ecuids: List[str],
    ) -> torch.Tensor:
        """
        Compute ECU verification reward.
        
        Verifies if student response mentions correct ECU name.
        Complements Teacher logprobs for precise term verification.
        
        Returns: [batch_size] ECU match scores
        """
        batch_size = len(student_responses)
        
        if self.teacher_agent is None:
            return torch.zeros(batch_size)
        
        ecu_scores = []
        for response, ecuid in zip(student_responses, ecuids):
            score = self.teacher_agent.compute_ecu_reward(
                student_response=response,
                ecuid=ecuid,
            )
            ecu_scores.append(score)
        
        return torch.tensor(ecu_scores)
    
    def compute_total_reward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        response_ids: torch.Tensor,
        mask: torch.Tensor,
        student_responses: List[str] = None,
        ecuids: List[str] = None,
        dtcs: List[str] = None,
    ) -> RewardOutput:
        """
        Compute total reward.
        
        total_reward = logit_reward + kg_reward_weight * kg_reward
        """
        batch_size, seq_len = response_ids.shape
        device = response_ids.device
        
        # 1. Compute Logit reward
        logit_reward, rev_kl = self.compute_logit_reward(
            student_logits, teacher_logits, response_ids, mask
        )
        
        # 2. Compute KG reward (if enabled)
        if student_responses and ecuids and dtcs and self.config.use_kg_reward:
            kg_reward_scalar, kg_info_list = self.compute_kg_reward(
                student_responses, ecuids, dtcs
            )
            kg_reward = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                valid_len = int(mask[i].sum().item())
                if valid_len > 0:
                    kg_reward[i, valid_len - 1] = kg_reward_scalar[i].item()
        else:
            kg_reward = torch.zeros(batch_size, seq_len, device=device)
            kg_info_list = [{}] * batch_size
        
        # 3. Compute total reward
        total_reward = logit_reward + kg_reward
        
        # 4. Reward scaling
        if self.config.reward_scaling:
            total_reward = total_reward / self.config.reward_scaling
        
        # 5. Reward clipping
        if self.config.cliprange_reward:
            total_reward = torch.clamp(
                total_reward,
                -self.config.cliprange_reward,
                self.config.cliprange_reward,
            )
        
        return RewardOutput(
            total_reward=total_reward,
            logit_reward=logit_reward,
            kg_reward=kg_reward,
            rev_kl=rev_kl,
            info={
                "kg_info_list": kg_info_list,
                "mean_logit_reward": logit_reward.sum() / mask.sum() if mask.sum() > 0 else 0,
                "mean_kg_reward": kg_reward.sum() / batch_size,
                "mean_rev_kl": rev_kl.sum() / mask.sum() if mask.sum() > 0 else 0,
            }
        )
    
    def compute_total_reward_v2(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        response_ids: torch.Tensor,
        mask: torch.Tensor,
        student_responses: List[str] = None,
        ecuids: List[str] = None,
        dtcs: List[str] = None,
        class_weights: List[float] = None,
        fd_weights: List[float] = None,
        sm_weights: List[float] = None,
    ) -> RewardOutput:
        """
        [New version] Compute reward directly from logprobs, supports FD/SM segmented weights.
        
        Key improvements:
        - Uses actual Teacher token scores instead of logits
        - Supports segmented weights: FD part uses fd_weight, SM part uses sm_weight
        - Low-frequency classes get higher reward weights
        
        Reward formula (segmented):
        R_t = weight(t) * log P_teacher(y_t)
        where weight(t) = fd_weight (if t in FD region) or sm_weight (if t in SM region)
        """
        batch_size, seq_len = response_ids.shape
        device = response_ids.device
        
        # Numerical stability
        student_logprobs = torch.nan_to_num(student_logprobs, nan=0.0, posinf=0.0, neginf=-20.0)
        teacher_logprobs = torch.nan_to_num(teacher_logprobs, nan=-5.0, posinf=0.0, neginf=-20.0)
        
        # 1. Compute reverse KL as distillation reward
        rev_kl = (teacher_logprobs - student_logprobs) * mask
        
        # Apply rev_kl weight
        rev_kl_weight = getattr(self.config, 'rev_kl_weight', 1.0)
        logit_reward = rev_kl * rev_kl_weight
        
        # 2. Compute KG reward (if enabled)
        if student_responses and ecuids and dtcs and self.config.use_kg_reward:
            kg_reward_scalar, kg_info_list = self.compute_kg_reward(
                student_responses, ecuids, dtcs
            )
            kg_reward = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                valid_len = int(mask[i].sum().item())
                if valid_len > 0:
                    kg_reward[i, valid_len - 1] = kg_reward_scalar[i].item()
        else:
            kg_reward = torch.zeros(batch_size, seq_len, device=device)
            kg_info_list = [{}] * batch_size
        
        # 2.5. Compute ECU verification reward
        ecu_reward = torch.zeros(batch_size, seq_len, device=device)
        ecu_reward_applied = False
        
        if student_responses and ecuids and getattr(self.config, 'use_ecu_reward', True):
            ecu_reward_applied = True
            ecu_scores = self.compute_ecu_reward(student_responses, ecuids)
            
            ecu_reward_weight = getattr(self.config, 'ecu_reward_weight', 5.0)
            ecu_penalty_weight = getattr(self.config, 'ecu_penalty_weight', -8.0)
            
            for i in range(batch_size):
                valid_len = int(mask[i].sum().item())
                if valid_len > 0:
                    score = ecu_scores[i].item()
                    
                    if score >= 0.5:
                        reward_value = ecu_reward_weight * (score - 0.5) * 2
                    else:
                        reward_value = ecu_penalty_weight * (0.5 - score) * 2
                    
                    ecu_reward[i, valid_len - 1] = reward_value
        
        # 3. Compute total reward
        total_reward = logit_reward + kg_reward + ecu_reward
        
        # 4. Apply segmented weights (FD and SM separately)
        segmented_weights_applied = False
        if fd_weights is not None and sm_weights is not None and student_responses is not None:
            segmented_weights_applied = True
            segment_weight_tensor = self._compute_segment_weights(
                student_responses=student_responses,
                fd_weights=fd_weights,
                sm_weights=sm_weights,
                seq_len=seq_len,
                mask=mask,
                device=device,
            )
            total_reward = total_reward * segment_weight_tensor
        
        # Legacy interface: if no segmented weights, use unified class_weights
        elif class_weights is not None:
            weight_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.mean()
            weight_tensor = torch.clamp(weight_tensor, min=0.5, max=3.0)
            total_reward = total_reward * weight_tensor.unsqueeze(1)
        
        # 5. Reward scaling
        if self.config.reward_scaling:
            total_reward = total_reward / self.config.reward_scaling
        
        # 6. Reward clipping
        if self.config.cliprange_reward:
            total_reward = torch.clamp(
                total_reward,
                -self.config.cliprange_reward,
                self.config.cliprange_reward,
            )
        
        return RewardOutput(
            total_reward=total_reward,
            logit_reward=logit_reward,
            kg_reward=kg_reward,
            rev_kl=rev_kl,
            info={
                "kg_info_list": kg_info_list,
                "mean_logit_reward": logit_reward.sum() / mask.sum() if mask.sum() > 0 else 0,
                "mean_kg_reward": kg_reward.sum() / batch_size,
                "mean_ecu_reward": ecu_reward.sum() / batch_size,
                "mean_rev_kl": rev_kl.sum() / mask.sum() if mask.sum() > 0 else 0,
                "segmented_weights_applied": segmented_weights_applied,
                "ecu_reward_applied": ecu_reward_applied,
            }
        )
    
    def _compute_segment_weights(
        self,
        student_responses: List[str],
        fd_weights: List[float],
        sm_weights: List[float],
        seq_len: int,
        mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute segmented weight tensor.
        
        JSON output format:
        {
          "FaultDescription": "...",   <- FD region
          "ServiceMeasures": "..."     <- SM region
        }
        
        Estimates token sequence proportions based on relative text lengths.
        """
        batch_size = len(student_responses)
        weight_tensor = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
        
        for i, response in enumerate(student_responses):
            try:
                parsed = json.loads(response)
                fd_text = parsed.get("FaultDescription", "")
                sm_text = parsed.get("ServiceMeasures", "")
                
                fd_len = len(fd_text)
                sm_len = len(sm_text)
                total_len = fd_len + sm_len + 1
                
                fd_ratio = fd_len / total_len
                
                valid_len = int(mask[i].sum().item())
                fd_token_len = int(valid_len * fd_ratio)
                
                fd_w = fd_weights[i]
                sm_w = sm_weights[i]
                mean_w = (fd_w + sm_w) / 2
                fd_w_norm = fd_w / mean_w
                sm_w_norm = sm_w / mean_w
                
                fd_w_norm = max(0.5, min(3.0, fd_w_norm))
                sm_w_norm = max(0.5, min(3.0, sm_w_norm))
                
                weight_tensor[i, :fd_token_len] = fd_w_norm
                weight_tensor[i, fd_token_len:valid_len] = sm_w_norm
                
            except (json.JSONDecodeError, KeyError, TypeError):
                weight_tensor[i, :] = 1.0
        
        return weight_tensor


def get_advantages(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    use_whitening: bool = True,
) -> torch.Tensor:
    """
    Compute advantage function.
    Uses cumulative discounted rewards with length normalization.
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device
    
    # Numerical stability
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Compute cumulative discounted rewards (backward)
    advantages = torch.zeros_like(rewards)
    running_reward = torch.zeros(batch_size, device=device)
    
    for t in reversed(range(seq_len)):
        running_reward = rewards[:, t] + gamma * running_reward * mask[:, t]
        advantages[:, t] = running_reward
    
    # Length normalization
    lens = mask.sum(dim=1, keepdim=True).clamp(min=1)
    advantages = advantages / lens
    
    # Whitening (standardization)
    if use_whitening:
        valid_advantages = advantages[mask.bool()]
        if len(valid_advantages) > 1:
            mean = valid_advantages.mean()
            std = valid_advantages.std().clamp(min=1e-8)
            advantages = (advantages - mean) / std
    
    # Final NaN/Inf check
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return advantages * mask
