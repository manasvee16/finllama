# Copilot Prompt:
# Load "meta-llama/Llama-2-7b-hf" using HuggingFace Transformers.
# Apply LoRA adapters using the peft library with:
# r=8, lora_alpha=32, target_modules=["q_proj","v_proj"].
# Freeze all base model parameters.
# Add a linear classification head mapping hidden_size to 3 sentiment classes.
# Return the PEFT-wrapped model.

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

def create_finllama_lora_model(base_model="meta-llama/Llama-2-7b-hf",
                              r=8,
                              lora_alpha=32,
                              target_modules=["q_proj", "v_proj"],
                              num_labels=3):
    """
    Load Llama-2-7B with LoRA adapters and classification head.
    
    Args:
        base_model: Base model name (default: meta-llama/Llama-2-7b-hf)
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: List of modules to apply LoRA
        num_labels: Number of sentiment classes
    
    Returns:
        PEFT-wrapped model ready for training
    """
    print(f"Loading {base_model} for sequence classification...")
    
    # Load base model for sequence classification
    base_model_obj = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    # Apply LoRA to base model
    print(f"Applying LoRA adapters (r={r}, lora_alpha={lora_alpha})...")
    model = get_peft_model(base_model_obj, lora_config)
    
    # Freeze all base model parameters - only LoRA parameters trainable
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    # Print parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / Total: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model

if __name__ == "__main__":
    model = create_finllama_lora_model()
    print("\nModel created successfully!")
    print(model)
