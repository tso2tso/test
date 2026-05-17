"""
Local Model Interface Module
Local model inference interface for loading and calling fine-tuned models
Supports both full model and LoRA adapter loading
Uses ModelScope for accelerated downloads (China region)
"""
import os
import json
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import ModelScope
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("[Warning] ModelScope not installed, will use HuggingFace download")

# Try to import PEFT
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[Warning] PEFT not installed, cannot load LoRA adapter")


class LocalModelInterface:
    """Local model inference interface"""
    
    def __init__(
        self,
        model_path: str,
        base_model_path: str = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        device: str = "cuda",
        torch_dtype: str = "float16",
        trust_remote_code: bool = True,
    ):
        """
        Initialize local model interface.
        
        Args:
            model_path: Model path (can be full model or LoRA adapter)
            base_model_path: Base model path (only needed for LoRA mode)
            max_new_tokens: Maximum generated tokens
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            device: Device (cuda/cpu)
            torch_dtype: Data type
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.device = device
        self.trust_remote_code = trust_remote_code
        
        # Set dtype
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32
        
        # Conversation history (for multi-turn dialogue)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Detect model type
        self.is_lora = self._check_is_lora()
        
        # Load model
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _check_is_lora(self) -> bool:
        """Check if this is a LoRA adapter"""
        adapter_config = os.path.join(self.model_path, "adapter_config.json")
        return os.path.exists(adapter_config)
    
    def _get_base_model_from_adapter(self) -> str:
        """Get base model name from adapter_config.json"""
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
                return config.get("base_model_name_or_path", None)
        return None
    
    def _download_model_with_modelscope(self, model_id: str) -> str:
        """
        Download model using ModelScope.
        
        Args:
            model_id: Model ID (e.g. Qwen/Qwen2.5-7B-Instruct)
            
        Returns:
            Local model path
        """
        if not MODELSCOPE_AVAILABLE:
            print(f"[LocalModel] ModelScope not available, returning original model_id: {model_id}")
            return model_id
        
        # Convert HuggingFace format to ModelScope format
        modelscope_id = model_id
        if "/" in model_id:
            org, name = model_id.split("/", 1)
            if org.lower() == "qwen":
                modelscope_id = f"qwen/{name}"
        
        print(f"[LocalModel] Downloading model with ModelScope: {modelscope_id}")
        try:
            local_path = snapshot_download(modelscope_id)
            print(f"[LocalModel] Model download complete: {local_path}")
            return local_path
        except Exception as e:
            print(f"[LocalModel] ModelScope download failed: {e}")
            print(f"[LocalModel] Trying original model_id: {model_id}")
            return model_id
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"[LocalModel] Loading model: {self.model_path}")
        print(f"[LocalModel] Device: {self.device}, dtype: {self.torch_dtype}")
        
        if self.is_lora:
            self._load_lora_model()
        else:
            self._load_full_model()
        
        print(f"[LocalModel] ✅ Model loaded successfully")
    
    def _load_full_model(self):
        """Load full model"""
        print(f"[LocalModel] Loading full model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            padding_side="left"
        )
        
        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
    
    def _load_lora_model(self):
        """Load LoRA adapter model"""
        if not PEFT_AVAILABLE:
            raise ImportError("Need to install PEFT to load LoRA adapter: pip install peft")
        
        print(f"[LocalModel] Detected LoRA adapter, loading...")
        
        # Get base model path
        base_model = self.base_model_path or self._get_base_model_from_adapter()
        if not base_model:
            raise ValueError("Cannot determine base model path, please specify base_model_path in config")
        
        print(f"[LocalModel] Base model: {base_model}")
        print(f"[LocalModel] LoRA adapter: {self.model_path}")
        
        # Use ModelScope to download base model (if remote model ID)
        if not os.path.exists(base_model):
            base_model = self._download_model_with_modelscope(base_model)
        
        # Load tokenizer (from adapter directory or base model)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                padding_side="left"
            )
            print(f"[LocalModel] Loaded tokenizer from adapter directory")
        except Exception as e:
            print(f"[LocalModel] Failed to load tokenizer from adapter directory: {e}")
            print(f"[LocalModel] Loading tokenizer from base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=self.trust_remote_code,
                padding_side="left"
            )
        
        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print(f"[LocalModel] Loading base model: {base_model}")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        
        # Load LoRA adapter
        print(f"[LocalModel] Loading LoRA adapter: {self.model_path}")
        self.model = PeftModel.from_pretrained(
            base_model_obj,
            self.model_path,
            torch_dtype=self.torch_dtype,
        )
        self.model.eval()
    
    def _build_prompt(self, ecu_id: str, dtc: str, trigger: str, timecondition: str) -> str:
        """
        Build prompt.
        
        Args:
            ecu_id: ECU ID
            dtc: DTC code
            trigger: Trigger condition
            timecondition: Time condition
            
        Returns:
            Built prompt
        """
        system_prompt = """You are a professional automotive fault diagnosis expert. Based on the given ECU ID, DTC (Diagnostic Trouble Code), and trigger conditions, provide accurate fault descriptions and service measures.
Please output strictly in JSON format with two fields: FaultDescription and ServiceMeasures."""

        user_prompt = f"ECUID: {ecu_id}, DTC: {dtc}, Trigger: {trigger}, TimeCondition: {timecondition}"
        
        return system_prompt, user_prompt
    
    def _parse_json_output(self, content: str) -> Dict[str, str]:
        """
        Parse model JSON output.
        
        Args:
            content: Model output content
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to find JSON portion
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
                return {
                    "FaultDescription": parsed.get("FaultDescription", parsed.get("fault_description", "")),
                    "RepairMeasures": parsed.get("RepairMeasures", parsed.get("ServiceMeasures", parsed.get("service_measures", "")))
                }
        except json.JSONDecodeError:
            pass
        
        # If parsing fails, return original content
        return {
            "FaultDescription": "",
            "RepairMeasures": content
        }
    
    @torch.no_grad()
    def predict(
        self,
        ecu_id: str,
        dtc: str,
        trigger: str,
        timecondition: str,
        temperature: Optional[float] = None
    ) -> tuple:
        """
        Make prediction.
        
        Args:
            ecu_id: ECU ID
            dtc: DTC code
            trigger: Trigger condition
            timecondition: Time condition
            temperature: Sampling temperature (optional, overrides default)
            
        Returns:
            (prediction result dict, raw output)
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Build prompt
        system_prompt, user_prompt = self._build_prompt(ecu_id, dtc, trigger, timecondition)
        
        # Use chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # If no chat template, manually build
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Encode input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        # Decode output (only take newly generated part)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse output
        parsed = self._parse_json_output(content)
        
        result = {
            "ecu_id": ecu_id,
            "dtc": dtc,
            "trigger": trigger,
            "timecondition": timecondition,
            "FaultDescription": parsed["FaultDescription"],
            "FaultDescription_gt": "",
            "RepairMeasures": parsed["RepairMeasures"],
            "RepairMeasures_gt": "",
        }
        
        raw_result = {
            "model": self.model_path,
            "content": content,
            "messages": messages
        }
        
        return result, raw_result
    
    def reset_conversation(self):
        """Reset conversation history (for multi-turn dialogue)"""
        self.conversation_history = []
    
    @torch.no_grad()
    def predict_with_history(
        self,
        ecu_id: str,
        dtc: str,
        trigger: str,
        timecondition: str,
        temperature: Optional[float] = None
    ) -> Dict[str, str]:
        """
        Prediction with conversation history (for multi-turn dialogue testing).
        
        Args:
            ecu_id: ECU ID
            dtc: DTC code
            trigger: Trigger condition
            timecondition: Time condition
            temperature: Sampling temperature
            
        Returns:
            Prediction result dictionary
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Build current turn prompt
        system_prompt, user_prompt = self._build_prompt(ecu_id, dtc, trigger, timecondition)
        
        # Build message list
        messages = []
        
        # Add system prompt if first turn
        if not self.conversation_history:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_prompt})
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Manually build
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
        
        # Encode input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192  # Multi-turn needs longer context
        ).to(self.model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": temp,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        # Decode output
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_prompt})
        self.conversation_history.append({"role": "assistant", "content": content})
        
        # Parse output
        parsed = self._parse_json_output(content)
        
        return {
            "ecu_id": ecu_id,
            "dtc": dtc,
            "trigger": trigger,
            "timecondition": timecondition,
            "FaultDescription": parsed["FaultDescription"],
            "FaultDescription_gt": "",
            "RepairMeasures": parsed["RepairMeasures"],
            "RepairMeasures_gt": "",
        }
    
    def re_predict_faultdescription(self, pred: Dict[str, str]) -> Dict[str, str]:
        """
        Regenerate fault description (for retry mechanism).
        
        Args:
            pred: Previous prediction result
            
        Returns:
            New prediction result
        """
        prompt = f"""
Your task is to provide a new fault description based on the given ECU ID, trigger, timecondition, DTC code, as well as the existing fault description.

Requirements:
1. The original fault description is incorrect or partially incorrect
2. Provide a more accurate fault description
3. When the corresponding information cannot be obtained, it can output 'unable to obtain corresponding information'.

The output format is as follows:
{{
    "new_FaultDescription": "[New brief explanation of the problem]. [New detailed description of the problem]"
}}

Here is the information you need to use:
    ECU_ID: {pred.get("ecu_id", "")}
    Trigger: {pred.get("trigger", "")}
    TimeCondition: {pred.get("timecondition", "")}
    DTC: {pred.get("dtc", "")}
    Original FaultDescription: {pred.get("FaultDescription", "")}

Please give your answer below:
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse new fault description
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
                new_fault = parsed.get("new_FaultDescription", content)
            else:
                new_fault = content
        except:
            new_fault = content
        
        return {
            "ecu_id": pred.get("ecu_id", ""),
            "dtc": pred.get("dtc", ""),
            "trigger": pred.get("trigger", ""),
            "timecondition": pred.get("timecondition", ""),
            "FaultDescription": new_fault,
            "RepairMeasures": pred.get("RepairMeasures", ""),
        }
    
    def re_predict_repairmeasures(self, pred: Dict[str, str], fault_judgment: str) -> Dict[str, str]:
        """
        Regenerate repair measures (for retry mechanism).
        
        Args:
            pred: Previous prediction result
            fault_judgment: Fault description judgment result
            
        Returns:
            New prediction result
        """
        fault_judge_bool = "true" if fault_judgment == "符合" else "false"
        
        prompt = f"""
Your task is to generate new and correct repair measures based on the ECU ID, Trigger, TimeCondition, DTC code.

Requirements:
1. Fault judgment is an important quantity. When its value is true, the generated new repair measures must comply with the given fault description; When its value is false, the given fault description must be ignored when generating new repair measures.
2. The original repair plan provided is incorrect or partially incorrect.

The output format is as follows:
{{
    "new_RepairMeasures": "[new repair measures text]"
}}

Here is the information you need to use:
    ECU_ID: {pred.get("ecu_id", "")}
    Trigger: {pred.get("trigger", "")}
    TimeCondition: {pred.get("timecondition", "")}
    DTC: {pred.get("dtc", "")}
    FaultDescription: {pred.get("FaultDescription", "")}
    Original RepairMeasures: {pred.get("RepairMeasures", "")}
    Fault Judgment: {fault_judge_bool}

Please give your answer below:
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse new repair measures
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                parsed = json.loads(json_str)
                new_repair = parsed.get("new_RepairMeasures", content)
            else:
                new_repair = content
        except:
            new_repair = content
        
        return {
            "ecu_id": pred.get("ecu_id", ""),
            "dtc": pred.get("dtc", ""),
            "trigger": pred.get("trigger", ""),
            "timecondition": pred.get("timecondition", ""),
            "FaultDescription": pred.get("FaultDescription", ""),
            "RepairMeasures": new_repair,
        }


if __name__ == "__main__":
    # Simple test
    import config
    
    if config.check_local_model():
        print("\nTesting local model inference...")
        model = LocalModelInterface(**config.LOCAL_MODEL_CONFIG)
        
        result, raw = model.predict(
            ecu_id="0087",
            dtc="5D0E",
            trigger="Voltage below 9 V and engine not in starting phase.",
            timecondition="500 ms"
        )
        
        print("\nPrediction result:")
        print(f"  FaultDescription: {result['FaultDescription']}")
        print(f"  RepairMeasures: {result['RepairMeasures']}")
    else:
        print("Local model not found, please check path config")
