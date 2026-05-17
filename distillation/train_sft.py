"""
SFT Training Script
Use LoRA to fine-tune Qwen3-8B for supervised learning
Goal: Teach the model to output JSON format for repair diagnostics
"""

import os
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG, MODEL_CONFIG, SFTConfig


def load_dataset(file_path: str) -> Dataset:
    """Load training data."""
    print(f"Loading dataset: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    print(f"Dataset size: {len(dataset)}")
    return dataset


def preprocess_function(
    examples: Dict,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
) -> Dict:
    """
    Preprocess function: Convert dialogue format to model input.
    Uses Qwen3 chat template.
    """
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    for messages in examples["messages"]:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels (only compute loss for assistant responses)
        labels = tokenized["input_ids"].copy()
        
        # Find assistant response start position
        assistant_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        
        input_ids = tokenized["input_ids"]
        assistant_start = -1
        for i in range(len(input_ids) - len(assistant_token), -1, -1):
            if input_ids[i:i+len(assistant_token)] == assistant_token:
                assistant_start = i + len(assistant_token)
                break
        
        if assistant_start > 0:
            labels[:assistant_start] = [-100] * assistant_start
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(tokenized["attention_mask"])
        model_inputs["labels"].append(labels)
    
    return model_inputs


def create_model_and_tokenizer(config: SFTConfig):
    """Create model and tokenizer."""
    print(f"Loading model: {config.model_name_or_path}")
    
    # Use ModelScope to download model (for better network access in China)
    from modelscope import snapshot_download
    model_id = config.model_name_or_path.replace("Qwen/", "qwen/")
    local_model_path = snapshot_download(model_id)
    print(f"Model path: {local_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config (optional, for memory savings)
    bnb_config = None
    if config.use_lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    if config.use_lora:
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, help="Model path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    args = parser.parse_args()

    # Load config
    config = SFTConfig()
    
    # Override config with command line args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_name_or_path:
        config.model_name_or_path = args.model_name_or_path
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(config)
    
    # Load dataset
    # 数据流: sft_balanced.json (balanced data for SFT stage)
    train_dataset = load_dataset(DATA_CONFIG["sft_data_path"])
    val_dataset = load_dataset(DATA_CONFIG["val_data_path"])
    
    # Preprocess data
    print("Preprocessing train set...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    
    print("Preprocessing val set...")
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.max_seq_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=["tensorboard"],
        logging_dir=os.path.join(config.output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    if config.deepspeed_config:
        training_args.deepspeed = config.deepspeed_config
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting SFT Training")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    
    if config.use_lora:
        # Save LoRA weights
        model.save_pretrained(os.path.join(config.output_dir, "lora_weights"))
        
        # Merge and save full model
        print("Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(config.output_dir, "merged_model"))
        tokenizer.save_pretrained(os.path.join(config.output_dir, "merged_model"))
    else:
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
    
    print(f"\nTraining complete! Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
